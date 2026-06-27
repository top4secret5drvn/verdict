import importlib

import verdict9
from cognitive_modules import ConceptFormationEngine


def test_verdict9_imports():
    assert importlib.import_module("verdict9") is verdict9


def test_parse_fact_and_evaluate_direct_predicate():
    kb = verdict9.KnowledgeBase()
    verdict9.parse_file("факт симптом(иван) = да причина факт.", kb)

    value, reason = verdict9.evaluate_predicate(kb, "симптом", ("иван",), {})

    assert value == verdict9.TRITS["да"]
    assert reason.kind == "факт"
    assert reason.args == ("иван",)


def test_belief_revision_prefers_higher_confidence_conflict():
    kb = verdict9.KnowledgeBase()
    verdict9.parse_file(
        """
        факт диагноз(иван) = да причина внешний(рентген) [confidence:0.95, источник:МРТ].
        факт диагноз(иван) = нет причина внешний(опрос) [confidence:0.40, источник:опрос].
        """,
        kb,
    )

    value, reason = verdict9.evaluate_predicate(kb, "диагноз", ("иван",), {})

    assert value == verdict9.TRITS["да"]
    assert reason.name == "диагноз"
    beliefs = kb.revision_engine.beliefs["диагноз(иван)"]
    defeated = [belief for belief in beliefs if belief.status == "defeated"]
    assert len(defeated) == 1
    assert defeated[0].value == verdict9.TRITS["нет"]


def test_concept_formation_generates_rule_for_consequent():
    kb = verdict9.KnowledgeBase()
    verdict9.parse_file(
        """
        факт симптом(иван) = да причина факт.
        факт температура(иван) = да причина факт.
        факт грипп(иван) = да причина факт.
        факт симптом(мария) = да причина факт.
        факт температура(мария) = да причина факт.
        факт грипп(мария) = да причина факт.
        факт симптом(олег) = да причина факт.
        факт температура(олег) = да причина факт.
        """,
        kb,
    )
    engine = ConceptFormationEngine(kb)
    engine.analyze_facts(min_support=2, min_confidence=0.5)
    engine.auto_generalize(top_n=1)

    assert any(rule.name == "грипп" for rule in kb.rules)
    value, reason = verdict9.evaluate_predicate(kb, "грипп", ("олег",), {})
    assert value == verdict9.TRITS["да"]
    assert reason.name == "грипп"
