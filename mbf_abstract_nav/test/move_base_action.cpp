// Concurrency test for mbf_abstract_nav::MoveBaseAction.
//
// MoveBaseAction drives its get_path, exe_path and recovery action clients from several threads: start and cancel
// run on the thread spinning the global callback queue, while replanning and the clients' callbacks run on others.
// actionlib::SimpleActionClient is not thread-safe, so MoveBaseAction must never use a client from two threads at
// once. This test keeps sending, replacing and canceling move_base goals, with replanning at 20 Hz and fake
// get_path, exe_path and recovery servers that answer within milliseconds. Unsynchronized client calls show up as
// a crash (SIGSEGV in actionlib), as actionlib errors about invalid goal handles, as a deadlock (the last goal
// never finishes) or as the same plan sent twice to exe_path.

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <actionlib/client/action_client.h>
#include <actionlib/server/action_server.h>
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/TransformStamped.h>
#include <mbf_msgs/ExePathAction.h>
#include <mbf_msgs/GetPathAction.h>
#include <mbf_msgs/MoveBaseAction.h>
#include <mbf_msgs/RecoveryAction.h>
#include <mbf_utility/navigation_utility.h>
#include <mbf_utility/robot_information.h>
#include <mbf_utility/types.h>
#include <ros/console.h>
#include <ros/ros.h>

#include "mbf_abstract_nav/MoveBaseFlexConfig.h"
#include "mbf_abstract_nav/move_base_action.h"

namespace
{
typedef actionlib::ActionClient<mbf_msgs::MoveBaseAction> MoveBaseClient;
typedef actionlib::ActionServer<mbf_msgs::MoveBaseAction> MoveBaseServer;

const std::string GLOBAL_FRAME = "map";
const std::string ROBOT_FRAME = "base_link";

// Counts the errors actionlib logs when a goal handle is reset while another thread uses it. They show the race
// even when it does not crash the test.
class ActionlibErrorCounter : public ros::console::LogAppender
{
public:
  ActionlibErrorCounter() : count_(0)
  {
  }

  virtual void log(ros::console::Level level, const char* str, const char*, const char*, int)
  {
    if (level >= ros::console::levels::Error &&
        (std::strstr(str, "on an inactive ClientGoalHandle") || std::strstr(str, "should not see invalid handles")))
    {
      ++count_;
    }
  }

  long count() const
  {
    return count_;
  }

private:
  boost::atomic<long> count_;
};

ActionlibErrorCounter* actionlib_errors = NULL;  // created in main

// Keeps the robot pose in the tf buffer fresh, as a localization would; MoveBaseAction needs it for every goal
class RobotPose
{
public:
  RobotPose() : stop_(false)
  {
    tf_.setUsingDedicatedThread(true);
    update();
    thread_ = boost::thread(&RobotPose::run, this);
  }

  ~RobotPose()
  {
    stop_ = true;
    thread_.join();
  }

  TF& tf()
  {
    return tf_;
  }

private:
  void update()
  {
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = GLOBAL_FRAME;
    transform.header.stamp = ros::Time::now();
    transform.child_frame_id = ROBOT_FRAME;
    transform.transform.rotation.w = 1.0;
    tf_.setTransform(transform, "move_base_action_test");
  }

  void run()
  {
    while (!stop_)
    {
      update();
      ros::WallDuration(0.02).sleep();
    }
  }

  TF tf_;
  boost::atomic<bool> stop_;
  boost::thread thread_;
};

// get_path, exe_path and recovery servers answering like MBF's, but within milliseconds. Unless calm, planning
// and control fail now and then, so MoveBaseAction also runs recovery behaviors.
class FakeNavigation
{
public:
  explicit FakeNavigation(ros::NodeHandle& nh)
    : calm_(false)
    , plans_(0)
    , exe_path_goals_(0)
    , duplicated_plans_(0)
    , last_plan_id_(-1.0)
    , get_path_rng_(11)
    , exe_path_rng_(13)
    , exe_path_goals_sub_(nh.subscribe("exe_path/goal", 1000, &FakeNavigation::exePathGoal, this))
    , get_path_(nh, "get_path", boost::bind(&FakeNavigation::getPath, this, _1), false)
    , exe_path_(nh, "exe_path", boost::bind(&FakeNavigation::exePath, this, _1), false)
    , recovery_(nh, "recovery", boost::bind(&FakeNavigation::recovery, this, _1), false)
  {
    get_path_.start();
    exe_path_.start();
    recovery_.start();
  }

  //! No more failures, and short executions, so a goal reliably succeeds
  void calmDown()
  {
    calm_ = true;
  }

  long plans() const
  {
    return plans_;
  }

  long exePathGoals() const
  {
    return exe_path_goals_;
  }

  long duplicatedPlans() const
  {
    return duplicated_plans_;
  }

private:
  void getPath(const mbf_msgs::GetPathGoalConstPtr&)
  {
    mbf_msgs::GetPathResult result;
    const int planning_ms = std::uniform_int_distribution<int>(0, 10)(get_path_rng_);
    for (int i = 0; i < planning_ms; ++i)
    {
      if (get_path_.isPreemptRequested())
      {
        result.outcome = mbf_msgs::GetPathResult::CANCELED;
        get_path_.setPreempted(result);
        return;
      }
      ros::WallDuration(0.001).sleep();
    }
    if (!calm_ && std::uniform_real_distribution<double>(0.0, 1.0)(get_path_rng_) < 0.05)
    {
      result.outcome = mbf_msgs::GetPathResult::FAILURE;
      get_path_.setAborted(result);
      return;
    }
    result.outcome = mbf_msgs::GetPathResult::SUCCESS;
    result.path.header.frame_id = GLOBAL_FRAME;
    result.path.header.stamp = ros::Time::now();
    result.path.poses.resize(2);
    for (size_t i = 0; i < result.path.poses.size(); ++i)
    {
      result.path.poses[i].header.frame_id = GLOBAL_FRAME;
      result.path.poses[i].pose.orientation.w = 1.0;
    }
    result.path.poses[0].pose.position.z = ++plans_;  // unique plan id, to detect a plan sent twice to exe_path
    result.path.poses[1].pose.position.x = 5.0;
    get_path_.setSucceeded(result);
  }

  void exePath(const mbf_msgs::ExePathGoalConstPtr&)
  {
    mbf_msgs::ExePathResult result;
    const int control_ms = calm_ ? 20 : std::uniform_int_distribution<int>(20, 400)(exe_path_rng_);
    const ros::WallTime end = ros::WallTime::now() + ros::WallDuration(control_ms / 1000.0);
    while (ros::WallTime::now() < end)
    {
      if (exe_path_.isPreemptRequested())
      {
        result.outcome = mbf_msgs::ExePathResult::CANCELED;
        exe_path_.setPreempted(result);
        return;
      }
      mbf_msgs::ExePathFeedback feedback;
      feedback.dist_to_goal = 5.0;  // far from the goal, so MoveBaseAction keeps replanning
      feedback.current_pose.header.frame_id = GLOBAL_FRAME;
      feedback.current_pose.pose.orientation.w = 1.0;
      exe_path_.publishFeedback(feedback);
      ros::WallDuration(0.01).sleep();
    }
    if (!calm_ && std::uniform_real_distribution<double>(0.0, 1.0)(exe_path_rng_) < 0.05)
    {
      result.outcome = mbf_msgs::ExePathResult::FAILURE;  // MoveBaseAction tries a recovery behavior
      exe_path_.setAborted(result);
      return;
    }
    result.outcome = mbf_msgs::ExePathResult::SUCCESS;
    exe_path_.setSucceeded(result);
  }

  void recovery(const mbf_msgs::RecoveryGoalConstPtr&)
  {
    ros::WallDuration(0.005).sleep();
    mbf_msgs::RecoveryResult result;
    result.outcome = mbf_msgs::RecoveryResult::SUCCESS;
    recovery_.setSucceeded(result);
  }

  // Sees every goal sent to exe_path, in order. The same plan twice in a row means that two threads handled the
  // same get_path result.
  void exePathGoal(const mbf_msgs::ExePathActionGoalConstPtr& msg)
  {
    ++exe_path_goals_;
    const double plan_id = msg->goal.path.poses.empty() ? -1.0 : msg->goal.path.poses[0].pose.position.z;
    if (plan_id >= 0.0 && plan_id == last_plan_id_)
    {
      ++duplicated_plans_;
    }
    last_plan_id_ = plan_id;
  }

  boost::atomic<bool> calm_;
  boost::atomic<long> plans_;
  boost::atomic<long> exe_path_goals_;
  boost::atomic<long> duplicated_plans_;
  double last_plan_id_;  // only used by exePathGoal
  std::mt19937 get_path_rng_;
  std::mt19937 exe_path_rng_;

  // declared last, so they are destroyed first, while the members used by their callbacks are still alive
  ros::Subscriber exe_path_goals_sub_;
  actionlib::SimpleActionServer<mbf_msgs::GetPathAction> get_path_;
  actionlib::SimpleActionServer<mbf_msgs::ExePathAction> exe_path_;
  actionlib::SimpleActionServer<mbf_msgs::RecoveryAction> recovery_;
};

mbf_msgs::MoveBaseGoal goalAt(double x)
{
  mbf_msgs::MoveBaseGoal goal;
  goal.target_pose.header.frame_id = GLOBAL_FRAME;
  goal.target_pose.pose.position.x = x;
  goal.target_pose.pose.orientation.w = 1.0;
  return goal;
}
}  // namespace

TEST(MoveBaseActionTest, concurrentGoalsAndCancels)
{
  ros::NodeHandle private_nh("~");
  ros::AsyncSpinner spinner(1);  // a single thread, as move_base_flex spins the global queue with ros::spin()
  spinner.start();

  RobotPose robot_pose;
  const ros::Duration tf_timeout(0.5);
  mbf_utility::RobotInformation robot_info(robot_pose.tf(), GLOBAL_FRAME, ROBOT_FRAME, tf_timeout);
  FakeNavigation navigation(private_nh);

  std::vector<std::string> recovery_behaviors;
  recovery_behaviors.push_back("recovery_a");
  recovery_behaviors.push_back("recovery_b");
  mbf_abstract_nav::MoveBaseAction move_base("move_base", robot_info, recovery_behaviors);
  mbf_abstract_nav::MoveBaseFlexConfig config = mbf_abstract_nav::MoveBaseFlexConfig::__getDefault__();
  config.planner_frequency = 20.0;
  config.oscillation_timeout = 0.0;
  config.recovery_enabled = true;
  move_base.reconfigure(config, 0);

  // forwards goals and cancels to MoveBaseAction, as AbstractNavigationServer does
  MoveBaseServer server(
      private_nh, "move_base", [&move_base](MoveBaseServer::GoalHandle goal_handle) { move_base.start(goal_handle); },
      [&move_base](MoveBaseServer::GoalHandle) { move_base.cancel(); }, false);
  server.start();

  MoveBaseClient client(private_nh, "move_base");
  ASSERT_TRUE(client.waitForActionServerToStart(ros::Duration(10.0)));

  // send, replace (without canceling) and cancel goals at random, as a mission executive may do
  std::mt19937 rng(7);
  MoveBaseClient::GoalHandle goal_handle;
  bool have_goal = false;
  const ros::WallTime end = ros::WallTime::now() + ros::WallDuration(15.0);
  while (ros::ok() && ros::WallTime::now() < end)
  {
    if (have_goal && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.1)
    {
      goal_handle.cancel();
    }
    else
    {
      goal_handle = client.sendGoal(goalAt(std::uniform_int_distribution<int>(1, 50)(rng)));
      have_goal = true;
    }
    ros::WallDuration(std::uniform_int_distribution<int>(5, 150)(rng) / 1000.0).sleep();
  }

  // MoveBaseAction must still work: cancel the last goal, then a new one must succeed
  navigation.calmDown();
  if (have_goal)
  {
    goal_handle.cancel();
  }
  ros::WallDuration(1.0).sleep();  // let MoveBaseAction cancel get_path, exe_path and recovery

  boost::mutex mutex;
  boost::condition_variable finished_condition;
  bool finished = false;
  actionlib::TerminalState terminal_state(actionlib::TerminalState::LOST);
  const MoveBaseClient::GoalHandle last_goal_handle =
      client.sendGoal(goalAt(10.0), [&](MoveBaseClient::GoalHandle handle) {
        if (handle.getCommState() == actionlib::CommState::DONE)
        {
          boost::lock_guard<boost::mutex> guard(mutex);
          terminal_state = handle.getTerminalState();
          finished = true;
          finished_condition.notify_all();
        }
      });
  {
    boost::unique_lock<boost::mutex> lock(mutex);
    finished_condition.wait_for(lock, boost::chrono::seconds(10), [&finished] { return finished; });
    EXPECT_TRUE(finished) << "the last goal did not finish within 10 s: MoveBaseAction is deadlocked";
    EXPECT_EQ(actionlib::TerminalState::SUCCEEDED, terminal_state.state_) << terminal_state.toString();
  }

  EXPECT_EQ(0, actionlib_errors->count()) << "actionlib errors: goal handles used while another thread reset them";
  EXPECT_EQ(0, navigation.duplicatedPlans()) << "plans sent twice in a row to exe_path";
  // the goals above did exercise MoveBaseAction
  EXPECT_GT(navigation.plans(), 100);
  EXPECT_GT(navigation.exePathGoals(), 20);

  spinner.stop();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "move_base_action_test");
  ros::NodeHandle nh;  // starts the node
  // Never deregistered and never deleted: rosconsole uses it until the process exits, and deregistering an
  // appender registered after roscpp's rosout appender crashes rosconsole 1.13 (Melodic)
  actionlib_errors = new ActionlibErrorCounter();
  ros::console::register_appender(actionlib_errors);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
