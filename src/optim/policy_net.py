import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Proste MLP operujące na ostatnim wymiarze wejścia.
    Wejście:  [..., input_features]
    Wyjście:  [..., 1]  (surowe 'mu' dla polityki; squash i sampling robimy w dOGR)

    Uwaga: dOGR zakłada output_features == 1 (bo potem robi .squeeze(-1)).
    """

    def __init__(
        self,
        input_features: int,
        output_features: int = 1,   # zostaw 1, patrz uwaga w docstringu
        hidden: int = 128,
        n_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert output_features == 1, "PolicyNet powinien zwracać 1 kanał (surowe mu)."

        layers = []
        dim_in = input_features
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(dim_in, hidden))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim_in = hidden

        # OSTATNIA WARSTWA: liniowa, BEZ aktywacji — zwraca surowe mu
        layers.append(nn.Linear(dim_in, output_features))
        self.net = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        # Lepsza inicjalizacja: wszystkie Linear oprócz ostatniego → Kaiming (ReLU),
        # ostatni Linear → małe wartości (stabilniej dla REINFORCE)
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        if not linear_layers:
            return
        *hidden_linears, last_linear = linear_layers

        for lin in hidden_linears:
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        nn.init.uniform_(last_linear.weight, a=-1e-3, b=1e-3)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
