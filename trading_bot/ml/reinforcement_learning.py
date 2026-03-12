"""
Reinforcement Learning Trading Agent.
Implements a sophisticated RL agent that learns optimal trading strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import structlog
from dataclasses import dataclass

from ..core.models import TradingSignal, TradeDecision, Position, OrderSide, OrderType
from ..strategy.base import Strategy, MarketContext, StrategyState
from .feature_engineering import FeatureEngineer, MarketFeatures

logger = structlog.get_logger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class RLConfig:
    """Configuration for RL agent."""
    state_size: int = 100
    action_size: int = 3  # 0: Hold, 1: Buy, 2: Sell
    hidden_size: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    reward_scaling: float = 1000.0


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RLTradingAgent(Strategy):
    """
    Reinforcement Learning Trading Agent.
    Uses Deep Q-Learning to learn optimal trading strategies.
    """
    
    def __init__(self, strategy_config, feature_engineer: FeatureEngineer):
        super().__init__(strategy_config)
        
        self.feature_engineer = feature_engineer
        self.config = RLConfig()
        
        # Initialize neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(
            self.config.state_size, 
            self.config.action_size, 
            self.config.hidden_size
        ).to(self.device)
        self.target_network = DQNNetwork(
            self.config.state_size, 
            self.config.action_size, 
            self.config.hidden_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.memory = ReplayBuffer(self.config.memory_size)
        
        # Training state
        self.epsilon = self.config.epsilon
        self.step_count = 0
        self.training_mode = True
        
        # Trading state
        self.current_state = None
        self.last_action = 0  # 0: Hold
        self.last_price = 0.0
        self.episode_reward = 0.0
        self.episode_trades = 0
        
        # Performance tracking
        self.total_episodes = 0
        self.win_episodes = 0
        self.total_reward = 0.0
        self.learning_history = []
        
        # Copy weights to target network
        self.update_target_network()
        
        logger.info(f"RL Trading Agent initialized with device: {self.device}")
    
    def get_state(self, symbol: str, context: MarketContext) -> Optional[np.ndarray]:
        """Get current state representation."""
        try:
            # Get features from feature engineer
            features = self.feature_engineer.extract_features(symbol)
            if not features:
                return None
            
            # Combine all features
            state_vector = []
            
            # Add price features
            state_vector.extend(list(features.price_features.values()))
            
            # Add technical features
            state_vector.extend(list(features.technical_features.values()))
            
            # Add volume features
            state_vector.extend(list(features.volume_features.values()))
            
            # Add volatility features
            state_vector.extend(list(features.volatility_features.values()))
            
            # Add microstructure features
            state_vector.extend(list(features.microstructure_features.values()))
            
            # Add temporal features
            state_vector.extend(list(features.temporal_features.values()))
            
            # Add regime features
            state_vector.extend(list(features.regime_features.values()))
            
            # Add position information
            if context.current_position:
                state_vector.extend([
                    float(context.current_position.quantity) / 1000.0,  # Normalized position size
                    float(context.current_position.unrealized_pnl) / 10000.0,  # Normalized P&L
                    1.0 if context.current_position.is_long else -1.0,  # Position direction
                ])
            else:
                state_vector.extend([0.0, 0.0, 0.0])
            
            # Add market context
            state_vector.extend([
                float(context.current_price) / 1000.0,  # Normalized price
                float(context.available_capital) / 100000.0,  # Normalized capital
                float(context.portfolio_exposure) / 100000.0,  # Normalized exposure
            ])
            
            # Pad or truncate to fixed size
            if len(state_vector) > self.config.state_size:
                state_vector = state_vector[:self.config.state_size]
            else:
                state_vector.extend([0.0] * (self.config.state_size - len(state_vector)))
            
            return np.array(state_vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if self.training_mode and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.config.action_size - 1)
        
        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def calculate_reward(self, action: int, context: MarketContext, next_context: MarketContext) -> float:
        """Calculate reward for the action taken."""
        try:
            reward = 0.0
            
            # Price change reward
            price_change = float(next_context.current_price - context.current_price)
            price_change_pct = price_change / float(context.current_price)
            
            # Action-based rewards
            if action == 1:  # Buy
                reward += price_change_pct * self.config.reward_scaling
            elif action == 2:  # Sell
                reward -= price_change_pct * self.config.reward_scaling
            # Hold (action == 0) gets no direct price reward
            
            # P&L reward
            if context.current_position and next_context.current_position:
                pnl_change = float(next_context.current_position.unrealized_pnl - context.current_position.unrealized_pnl)
                reward += pnl_change / 100.0  # Scaled P&L reward
            
            # Risk penalty
            if next_context.current_position:
                position_size = abs(float(next_context.current_position.quantity))
                max_position = 1000  # Maximum allowed position
                if position_size > max_position:
                    reward -= (position_size - max_position) * 0.1
            
            # Transaction cost penalty
            if action != 0:  # If not holding
                reward -= 0.1  # Small transaction cost
            
            # Volatility penalty for large positions
            if next_context.current_position:
                position_value = abs(float(next_context.current_position.quantity)) * float(next_context.current_price)
                if position_value > 50000:  # Large position
                    reward -= 0.05
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def should_enter_trade(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Determine if should enter trade using RL agent."""
        try:
            # Get current state
            state = self.get_state(signal.symbol, context)
            if state is None:
                return False
            
            # Select action
            action = self.select_action(state)
            
            # Store state for learning
            self.current_state = state
            self.last_action = action
            self.last_price = float(context.current_price)
            
            # Convert action to trading decision
            return action != 0  # Buy (1) or Sell (2), not Hold (0)
            
        except Exception as e:
            logger.error(f"Error in RL should_enter_trade: {e}")
            return False
    
    def calculate_position_size(self, context: MarketContext) -> int:
        """Calculate position size based on RL action and confidence."""
        try:
            if self.current_state is None:
                return 0
            
            # Get Q-values for current state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_confidence = torch.softmax(q_values, dim=1).max().item()
            
            # Base position size
            base_capital = context.available_capital * self.allocation
            base_position_value = base_capital * Decimal(str(self.position_size_pct))
            base_quantity = int(base_position_value / context.current_price)
            
            # Adjust based on confidence
            confidence_multiplier = min(action_confidence * 2, 1.5)  # Max 1.5x
            final_quantity = int(base_quantity * confidence_multiplier)
            
            # Ensure minimum and maximum limits
            final_quantity = max(final_quantity, 1)
            final_quantity = min(final_quantity, 1000)  # Max position limit
            
            return final_quantity
            
        except Exception as e:
            logger.error(f"Error calculating RL position size: {e}")
            return 0
    
    def get_exit_conditions(self, context: MarketContext) -> List:
        """Get exit conditions (RL agent manages exits internally)."""
        # RL agent will decide exits through its action selection
        return []
    
    def update_experience(self, symbol: str, next_context: MarketContext, done: bool = False):
        """Update experience and train the agent."""
        try:
            if self.current_state is None:
                return
            
            # Get next state
            next_state = self.get_state(symbol, next_context)
            if next_state is None:
                return
            
            # Calculate reward
            current_context = MarketContext(
                symbol=symbol,
                current_price=Decimal(str(self.last_price)),
                available_capital=next_context.available_capital,
                portfolio_exposure=next_context.portfolio_exposure
            )
            
            reward = self.calculate_reward(self.last_action, current_context, next_context)
            self.episode_reward += reward
            
            # Store experience
            experience = Experience(
                state=self.current_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            self.memory.push(experience)
            
            # Train if enough experiences
            if len(self.memory) >= self.config.batch_size and self.training_mode:
                self.train()
            
            # Update state
            self.current_state = next_state
            self.step_count += 1
            
            # Update target network periodically
            if self.step_count % self.config.target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            if self.epsilon > self.config.epsilon_min:
                self.epsilon *= self.config.epsilon_decay
            
            # End episode handling
            if done:
                self.end_episode()
            
        except Exception as e:
            logger.error(f"Error updating RL experience: {e}")
    
    def train(self):
        """Train the DQN network."""
        try:
            if len(self.memory) < self.config.batch_size:
                return
            
            # Sample batch
            batch = self.memory.sample(self.config.batch_size)
            
            # Prepare batch data
            states = torch.FloatTensor([e.state for e in batch]).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
            dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Log training metrics
            self.learning_history.append({
                'step': self.step_count,
                'loss': loss.item(),
                'epsilon': self.epsilon,
                'avg_q_value': current_q_values.mean().item()
            })
            
        except Exception as e:
            logger.error(f"Error training RL agent: {e}")
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def end_episode(self):
        """Handle end of trading episode."""
        self.total_episodes += 1
        self.total_reward += self.episode_reward
        
        if self.episode_reward > 0:
            self.win_episodes += 1
        
        # Log episode results
        win_rate = self.win_episodes / self.total_episodes if self.total_episodes > 0 else 0
        avg_reward = self.total_reward / self.total_episodes if self.total_episodes > 0 else 0
        
        logger.info(f"RL Episode {self.total_episodes} completed: "
                   f"Reward: {self.episode_reward:.2f}, "
                   f"Win Rate: {win_rate:.2%}, "
                   f"Avg Reward: {avg_reward:.2f}, "
                   f"Epsilon: {self.epsilon:.3f}")
        
        # Reset episode state
        self.episode_reward = 0.0
        self.episode_trades = 0
        self.current_state = None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'total_episodes': self.total_episodes,
                'config': self.config
            }, filepath)
            logger.info(f"RL model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.total_episodes = checkpoint['total_episodes']
            logger.info(f"RL model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
        logger.info(f"RL agent training mode: {training}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL agent performance metrics."""
        win_rate = self.win_episodes / self.total_episodes if self.total_episodes > 0 else 0
        avg_reward = self.total_reward / self.total_episodes if self.total_episodes > 0 else 0
        
        return {
            'total_episodes': self.total_episodes,
            'win_episodes': self.win_episodes,
            'win_rate': win_rate,
            'total_reward': self.total_reward,
            'avg_reward': avg_reward,
            'current_epsilon': self.epsilon,
            'training_steps': self.step_count,
            'memory_size': len(self.memory),
            'recent_losses': [h['loss'] for h in self.learning_history[-10:]] if self.learning_history else []
        }