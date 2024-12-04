
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent

# ...existing code...

def initialize_agent(
    action_dim,
    state_dim,  # Ensure this matches
    use_gpu,
    mcts_simulations_per_move,
    c_puct,
    load_model=True
) -> AlphaZeroAgent:
    agent = AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,        
        mcts_simulations_per_move=mcts_simulations_per_move,        
        c_puct=c_puct,        
        load_model=load_model
    )
    return agent

# ...existing code...