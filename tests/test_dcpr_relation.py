import torch
import torch.nn as nn

import methods.dcpr as dcpr


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _build_network(monkeypatch):
    monkeypatch.setattr(dcpr, "get_backbone", lambda _: _TinyBackbone())
    network = dcpr.DCPRNetwork(
        backbone_name="tiny",
        num_classes=3,
        num_source_domains=2,
        bottleneck_dim=0,
        relation_temperature=0.2,
        relation_space_mode="domain_relative",
    )
    network.relation_router.src_proto_inited.fill_(True)
    network.relation_router.src_domain_center_inited.fill_(True)
    network.relation_router.src_domain_centers.copy_(
        torch.tensor(
            [
                [0.2, 0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0, 0.0],
            ]
        )
    )
    network.relation_router.src_prototypes.copy_(
        torch.tensor(
            [
                [
                    [1.2, 0.0, 0.0, 0.0],
                    [0.2, 1.0, 0.0, 0.0],
                    [0.2, 0.0, 1.0, 0.0],
                ],
                [
                    [0.8, 0.0, 0.0, 0.0],
                    [-0.2, 1.0, 0.0, 0.0],
                    [-0.2, 0.0, 1.0, 0.0],
                ],
            ]
        )
    )
    with torch.no_grad():
        network.adaptive_classifier.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    return network


def test_target_center_changes_relation_but_not_shared_classifier(monkeypatch):
    network = _build_network(monkeypatch)
    features = torch.tensor(
        [
            [1.0, 0.3, -0.2, 0.1],
            [-0.1, 0.8, 0.2, -0.2],
        ]
    )

    uncentered_logits, uncentered_aux = network.forward_relation_logits(h_shared=features)
    network.update_target_center(features + 0.4, momentum=0.98)
    centered_logits, centered_aux = network.forward_relation_logits(
        h_shared=features,
        use_target_center=True,
    )

    assert torch.allclose(uncentered_logits, centered_logits)
    assert not torch.allclose(
        uncentered_aux["h_relation"],
        centered_aux["h_relation"],
    )


def test_classifier_margin_rank_is_class_count_normalized(monkeypatch):
    network = _build_network(monkeypatch)
    prototypes = torch.tensor(
        [
            [
                [1.00, 0.00, 0.00, 0.00],
                [0.70, 0.71, 0.00, 0.00],
                [0.30, 0.00, 0.95, 0.00],
            ],
            [
                [1.00, 0.00, 0.00, 0.00],
                [0.70, 0.71, 0.00, 0.00],
                [0.30, 0.00, 0.95, 0.00],
            ],
        ]
    )
    valid = torch.ones(2, 3, dtype=torch.bool)
    solver = object.__new__(dcpr.DCPRSolver)
    solver.num_classes = 3
    solver.ambiguity_relation_boost = 0.5
    solver.class_ambiguity_weights = torch.zeros(3)

    solver._update_class_ambiguity(network, valid, prototypes)

    assert torch.allclose(
        solver.class_ambiguity_weights,
        torch.tensor([0.0, 1.0, 0.5]),
    )
    uniform = torch.full((1, 3), 1.0 / 3.0)
    assert torch.allclose(solver._ambiguity_sample_weights(uniform), torch.ones(1))
    assert torch.allclose(
        solver._ambiguity_sample_weights(torch.tensor([[0.0, 1.0, 0.0]])),
        torch.tensor([1.5]),
    )


def test_consistency_targets_select_distinct_distributions():
    solver = object.__new__(dcpr.DCPRSolver)
    aux = {
        "class_probs": torch.tensor([[0.7, 0.3]]),
        "prototype_class_probs": torch.tensor([[0.4, 0.6]]),
        "node_mass": torch.tensor([[[0.6, 0.1], [0.1, 0.2]]]),
    }

    solver.consistency_target = "relation"
    assert torch.allclose(
        solver._consistency_distribution(aux),
        torch.tensor([[0.6, 0.1, 0.1, 0.2]]),
    )

    solver.consistency_target = "classification"
    assert solver._consistency_distribution(aux) is aux["class_probs"]

    solver.consistency_target = "class_only"
    assert solver._consistency_distribution(aux) is aux["prototype_class_probs"]


def test_routing_selection_prefers_class_conditioned_route_diversity():
    probs = torch.tensor([[0.6, 0.4], [0.6, 0.4]])
    routes = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.95, 0.05], [0.05, 0.95]],
        ]
    )

    assert dcpr._select_routing_indices(probs, routes, 1, 2) == [1]


def test_ambiguous_selection_keeps_source_errors_corrected_by_dcpr():
    source_probs = torch.tensor(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
        ]
    )
    dcpr_probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 0])
    cases = dcpr._select_ambiguous_cases(
        source_probs,
        dcpr_probs,
        labels,
        torch.tensor([1.0, 0.5, 0.0]),
        num_pairs=2,
        samples_per_pair=1,
    )

    assert {(case["true_class"], case["confused_class"]) for case in cases} == {
        (0, 1),
        (1, 2),
    }
