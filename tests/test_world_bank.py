from crisis_ews.data.world_bank import WorldBankCollector


def test_parse_world_bank_payload_preserves_observed_value_schema() -> None:
    payloads = [
        [
            {"page": 1, "pages": 1},
            [
                {
                    "countryiso3code": "AAA",
                    "country": {"value": "A"},
                    "date": "2001",
                    "value": 1.5,
                },
                {
                    "countryiso3code": "AAA",
                    "country": {"value": "A"},
                    "date": "2000",
                    "value": None,
                },
            ],
        ]
    ]
    frame = WorldBankCollector.parse_indicator_payload(payloads, "TEST.CODE")
    assert frame.to_dict("records") == [
        {"country_code": "AAA", "country_name": "A", "year": 2001, "indicator_code": "TEST.CODE", "value": 1.5}
    ]


def test_parse_accepts_direct_single_page_api_payload() -> None:
    payload = [
        {"page": 1, "pages": 1},
        [{"countryiso3code": "AAA", "country": {"value": "A"}, "date": "2001", "value": 1.5}],
    ]
    frame = WorldBankCollector.parse_indicator_payload(payload, "TEST.CODE")
    assert len(frame) == 1
    assert frame.loc[0, "year"] == 2001
