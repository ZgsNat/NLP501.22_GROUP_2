from app.eval import defense_numbers as dn


def test_every_deck_number_is_still_in_its_source():
    assert dn.verify() == []


def test_report_cites_a_line_for_every_number():
    text = dn.report()
    assert "không tìm thấy" not in text
    assert all(n.value in text for n in dn.NUMS)


def test_every_number_has_its_data_and_a_worked_calculation():
    assert all(n.data and n.calc for n in dn.NUMS)
    assert len(dn.WORKED) == len(dn.NUMS)


def test_footnotes_group_numbers_by_slide():
    notes = dn.footnotes()
    assert sum(len(v) for v in notes.values()) == len(dn.NUMS)
    assert all(row["line"] for rows in notes.values() for row in rows)
