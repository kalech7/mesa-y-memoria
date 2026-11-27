"""E1: same snapshots, no SELECT, interaction or memory updates."""
from copy import deepcopy
from ..scoring.evidence import score_response
from ..protocol import assert_split_allowed
from ..artifacts import canonical


def evaluate_snapshots(episodes, policy_factory, *, method="B4", root=None):
    result = []
    for episode in episodes:
        assert_split_allowed(episode["split"], root=root)
        all_records = deepcopy(episode["initial_memory"])
        relevant = [r for r in all_records if r["attribute"] == episode["current_query"]["attribute"]]
        latest = [max(enumerate(relevant), key=lambda pair: (pair[1].get("recorded_at", ""), pair[0]))[1]] if relevant else []
        for condition, memory in (("O1", latest), ("O2", all_records)):
            observation = {"memory": memory, "user_messages": deepcopy(episode["user_messages"]),
                           "query": deepcopy(episode["current_query"]),
                           "reuse_regime": "Habrá una consulta posterior sin canal de aclaración."}
            policy = policy_factory()
            try:
                response = policy.respond(deepcopy(observation))
            except (ValueError, KeyError, TypeError):
                response = {}
            score = score_response(response, observation, episode["truth"]["current_value"])
            result.append({"method": method, "seed": 101, "root_id": episode["root_id"],
                           "latent_world_id": episode["latent_world_id"], "realization_id": episode["realization_id"],
                           "scenario_family": episode["scenario_family"], "split": episode["split"],
                           "observation_condition": condition, "C2": score["success"], "score": score,
                           "input_characters": len(canonical(observation)), "silent_truncation": False,
                           "response": response})
    return result
