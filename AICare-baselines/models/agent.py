from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.utils import get_last_visit


class AgentLayer(nn.Module):
    """Dr. Agent layer.

    Paper: Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked second opinions. JAMIA.

    This layer is used in the Dr. Agent model. But it can also be used as a
    standalone layer.

    Args:
        lab_dim: dynamic feature size.
        demo_dim: static feature size, if 0, then no static feature is used.
        cell: rnn cell type. Default is "gru".
        use_baseline: whether to use baseline for the RL agent. Default is True.
        n_actions: number of historical visits to choose. Default is 10.
        n_units: number of hidden units in each agent. Default is 64.
        fusion_dim: number of hidden units in the final representation. Default is 128.
        hidden_dim: number of hidden units in the rnn. Default is 128.
        dropout: dropout rate. Default is 0.5.
        lamda: weight for the agent selected hidden state and the current hidden state. Default is 0.5.
    """

    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        cell: str = "gru",
        use_baseline: bool = True,
        n_actions: int = 10,
        n_units: int = 64,
        hidden_dim: int = 128,
        dropout: int = 0.5,
        lamda: int = 0.5,
    ):
        super(AgentLayer, self).__init__()

        if cell not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for cell.")

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim
        # self.n_output = n_output
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = hidden_dim
        self.demo_dim = demo_dim

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        self.agent1_fc1 = nn.Linear(self.hidden_dim + self.demo_dim, self.n_units)
        self.agent2_fc1 = nn.Linear(self.lab_dim + self.demo_dim, self.n_units)
        self.agent1_fc2 = nn.Linear(self.n_units, self.n_actions)
        self.agent2_fc2 = nn.Linear(self.n_units, self.n_actions)
        if use_baseline == True:
            self.agent1_value = nn.Linear(self.n_units, 1)
            self.agent2_value = nn.Linear(self.n_units, 1)

        if self.cell == "lstm":
            self.rnn = nn.LSTMCell(self.lab_dim, self.hidden_dim)
        else:
            self.rnn = nn.GRUCell(self.lab_dim, self.hidden_dim)

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        if dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=dropout)

        if self.demo_dim > 0:
            self.init_h = nn.Linear(self.demo_dim, self.hidden_dim)
            self.init_c = nn.Linear(self.demo_dim, self.hidden_dim)
            self.fusion = nn.Linear(self.hidden_dim + self.demo_dim, self.fusion_dim)
        # self.output = nn.Linear(self.fusion_dim, self.n_output)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def choose_action(self, observation, agent=1):
        observation = observation.detach()

        if agent == 1:
            result_fc1 = self.agent1_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent1_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent1_value(result_fc1)
                self.agent1_baseline.append(result_value)
        else:
            result_fc1 = self.agent2_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent2_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent2_value(result_fc1)
                self.agent2_baseline.append(result_value)

        probs = self.softmax(result_fc2)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()

        if agent == 1:
            self.agent1_entropy.append(m.entropy())
            self.agent1_action.append(actions.unsqueeze(-1))
            self.agent1_prob.append(m.log_prob(actions))
        else:
            self.agent2_entropy.append(m.entropy())
            self.agent2_action.append(actions.unsqueeze(-1))
            self.agent2_prob.append(m.log_prob(actions))

        return actions.unsqueeze(-1)

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_output: a tensor of shape [batch size, hidden_dim] representing the
                patient embedding.
            output: a tensor of shape [batch size, sequence len, hidden_dim] representing the patient embedding at each time step.
        """
        # rnn will only apply dropout between layers

        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        if self.demo_dim > 0:
            cur_h = self.init_h(static)
            if self.cell == "lstm":
                cur_c = self.init_c(static)
        else:
            cur_h = torch.zeros(
                batch_size, self.hidden_dim, dtype=torch.float32, device=x.device
            )
            if self.cell == "lstm":
                cur_c = torch.zeros(
                    batch_size, self.hidden_dim, dtype=torch.float32, device=x.device
                )

        h = []
        for cur_time in range(time_step):
            cur_input = x[:, cur_time, :]

            if cur_time == 0:
                obs_1 = cur_h
                obs_2 = cur_input

                if self.demo_dim > 0:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                self.choose_action(obs_1, 1).long()
                self.choose_action(obs_2, 2).long()

                observed_h = (
                    torch.zeros_like(cur_h, dtype=torch.float32)
                    .view(-1)
                    .repeat(self.n_actions)
                    .view(self.n_actions, batch_size, self.hidden_dim)
                )
                action_h = cur_h
                if self.cell == "lstm":
                    observed_c = (
                        torch.zeros_like(cur_c, dtype=torch.float32)
                        .view(-1)
                        .repeat(self.n_actions)
                        .view(self.n_actions, batch_size, self.hidden_dim)
                    )
                    action_c = cur_c

            else:
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), 0)

                obs_1 = observed_h.mean(dim=0)
                obs_2 = cur_input

                if self.demo_dim > 0:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                act_idx1 = self.choose_action(obs_1, 1).long()
                act_idx2 = self.choose_action(obs_2, 2).long()
                batch_idx = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)
                action_h1 = observed_h[act_idx1, batch_idx, :].squeeze(1)
                action_h2 = observed_h[act_idx2, batch_idx, :].squeeze(1)
                action_h = (action_h1 + action_h2) / 2
                if self.cell == "lstm":
                    observed_c = torch.cat((observed_c[1:], cur_c.unsqueeze(0)), 0)
                    action_c1 = observed_c[act_idx1, batch_idx, :].squeeze(1)
                    action_c2 = observed_c[act_idx2, batch_idx, :].squeeze(1)
                    action_c = (action_c1 + action_c2) / 2

            if self.cell == "lstm":
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                weighted_c = self.lamda * action_c + (1 - self.lamda) * cur_c
                rnn_state = (weighted_h, weighted_c)
                cur_h, cur_c = self.rnn(cur_input, rnn_state)
            else:
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                cur_h = self.rnn(cur_input, weighted_h)
            h.append(cur_h)

        h = torch.stack(h, dim=1)

        if self.demo_dim > 0:
            static = static.unsqueeze(1).repeat(1, time_step, 1)
            h = torch.cat((h, static), dim=2)
            h = self.fusion(h)

        last_out = get_last_visit(h, mask)

        if self.dropout > 0.0:
            last_out = self.nn_dropout(last_out)
        return last_out, h

class Agent(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        cell: str = "gru",
        use_baseline: bool = True,
        n_actions: int = 10,
        n_units: int = 64,
        hidden_dim: int = 128,
        dropout: int = 0.0,
        lamda: int = 0.5,
        **kwargs
    ):
        super(Agent, self).__init__()

        if cell not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for cell.")

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim
        # self.n_output = n_output
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = hidden_dim
        self.demo_dim = demo_dim
        
        self.agent_encoder = AgentLayer(lab_dim, demo_dim, cell, use_baseline, n_actions, n_units, hidden_dim, dropout, lamda,)
    
    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        # batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out, _ = self.agent_encoder(cur_x, static, cur_mask)
        #     out[:, cur_time, :] = cur_out
        _, out = self.agent_encoder(x, static, mask)
        return out[:, -1, :]
