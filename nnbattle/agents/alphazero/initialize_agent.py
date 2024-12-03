
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent

# ...existing code...

def initialize_agent(
    action_dim=7,
    state_dim=3,  # Ensure this matches
    use_gpu=True,
    num_simulations=10,
    c_puct=1.4,
    load_model=True
) -> AlphaZeroAgent:
    agent = AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        use_gpu=use_gpu,        
        num_simulations=num_simulations,        
        c_puct=c_puct,        
        load_model=load_model
    )
    # Verify that the model is initialized
    if agent.model is None:
        logger.error("Agent model is not initialized in initialize_agent.")
        raise AttributeError("Agent model is not initialized.")
    logger.info("Agent initialized with a valid model.")
    return agent

# ...existing code...