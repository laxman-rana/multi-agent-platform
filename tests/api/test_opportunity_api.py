import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.integrations.company_resolver import resolve_company_names
from src.integrations.whatsapp import extract_tickers_from_text


class OpportunityApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("src.api.app.run_opportunity_scan")
    def test_scan_endpoint_returns_normalized_unique_tickers(self, run_scan_mock):
        run_scan_mock.return_value = (
            ["AAPL", "MSFT"],
            [
                {
                    "ticker": "AAPL",
                    "confidence": "high",
                    "reason": "Sample recommendation",
                    "score": 8,
                }
            ],
        )

        response = self.client.post(
            "/api/v1/opportunity/scan",
            json={"tickers": ["aapl", " msft ", "AAPL"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "tickers": ["AAPL", "MSFT"],
                "opportunity_count": 1,
                "opportunities": [
                    {
                        "ticker": "AAPL",
                        "confidence": "high",
                        "reason": "Sample recommendation",
                        "score": 8,
                    }
                ],
            },
        )
        run_scan_mock.assert_called_once_with(["aapl", " msft ", "AAPL"])

    def test_scan_endpoint_rejects_empty_tickers(self):
        response = self.client.post(
            "/api/v1/opportunity/scan",
            json={"tickers": [" ", ""]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "Provide at least one non-empty ticker symbol.",
        )

    @patch("src.api.app.run_supervisor_query")
    def test_assistant_query_uses_supervisor_service(self, run_supervisor_mock):
        run_supervisor_mock.return_value = {
            "message": "Should I buy Microsoft?",
            "resolved_tickers": ["MSFT"],
            "opportunity_count": 1,
            "opportunities": [
                {
                    "ticker": "MSFT",
                    "confidence": "high",
                    "reason": "Durable quality profile.",
                    "score": 8,
                }
            ],
            "reply_text": "Microsoft looks attractive right now.",
            "worker_results": {"scan_opportunities": "[]"},
        }

        response = self.client.post(
            "/api/v1/assistant/query",
            json={"message": "Should I buy Microsoft?"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["resolved_tickers"], ["MSFT"])
        self.assertEqual(body["opportunity_count"], 1)
        self.assertIn("Microsoft", body["reply_text"])
        run_supervisor_mock.assert_called_once_with(message="Should I buy Microsoft?", model=None)

    @patch("src.api.app.run_supervisor_query")
    def test_whatsapp_webhook_uses_supervisor_service(self, run_supervisor_mock):
        run_supervisor_mock.return_value = {
            "message": "recommend microsoft",
            "resolved_tickers": ["MSFT"],
            "opportunity_count": 1,
            "opportunities": [
                {
                    "ticker": "MSFT",
                    "confidence": "high",
                    "reason": "Durable quality profile.",
                    "score": 8,
                }
            ],
            "reply_text": "Microsoft looks attractive right now.",
            "worker_results": {"scan_opportunities": "[]"},
        }

        response = self.client.post(
            "/webhooks/whatsapp",
            json={"message": "recommend microsoft"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["resolved_tickers"], ["MSFT"])
        self.assertEqual(body["opportunity_count"], 1)
        self.assertIn("Microsoft", body["reply_text"])
        run_supervisor_mock.assert_called_once_with(message="recommend microsoft")

    def test_whatsapp_webhook_handles_empty_message(self):
        response = self.client.post("/webhooks/whatsapp", json={"message": ""})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["resolved_tickers"], [])
        self.assertEqual(body["opportunity_count"], 0)
        self.assertIn("Should I buy Microsoft", body["reply_text"])


class WhatsAppExtractionTests(unittest.TestCase):
    def test_extract_tickers_from_text(self):
        self.assertEqual(
            extract_tickers_from_text("Should I buy nvda, msft and $aapl today?"),
            ["NVDA", "MSFT", "AAPL"],
        )

    @patch("src.integrations.company_resolver._search_yahoo")
    def test_resolve_company_names_uses_yahoo_result(self, search_mock):
        search_mock.return_value = [
            {
                "symbol": "MSFT",
                "quoteType": "EQUITY",
                "shortname": "Microsoft Corporation",
            }
        ]
        self.assertEqual(resolve_company_names("give me recommendation for microsoft"), ["MSFT"])

    @patch("src.integrations.company_resolver._search_yahoo", side_effect=RuntimeError("network down"))
    def test_resolve_company_names_returns_empty_when_yahoo_fails(self, _search_mock):
        self.assertEqual(resolve_company_names("give me recommendation for microfsoft"), [])


if __name__ == "__main__":
    unittest.main()
