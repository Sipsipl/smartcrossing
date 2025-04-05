import time
from rl.environment import SmartCrossingEnv
from rl.agent import DQNAgent

def test_episode_speed():
    """Test how long an episode takes without rendering"""
    # Create environment without rendering
    env = SmartCrossingEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Run one episode
    start_time = time.time()
    state, _ = env.reset()
    done = False
    steps = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        steps += 1
    
    duration = time.time() - start_time
    print(f"Episode completed in {duration:.2f} seconds")
    print(f"Total steps: {steps}")
    print(f"Average time per step: {(duration/steps)*1000:.2f} ms")
    
    env.close()

if __name__ == "__main__":
    test_episode_speed() 