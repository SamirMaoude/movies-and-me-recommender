"""Check deployed model identity without a running service."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.check_api import check, main


class APISmokeIdentityTests(unittest.TestCase):
    def mock_http(self, run_id):
        identity = {"model_uri": "/models/als", "run_id": run_id}

        def respond(request, timeout):
            path = request.full_url.removeprefix("http://api.test")
            status = 200
            if path == "/health":
                body = {"status": "ok"}
            elif path == "/model/info":
                body = identity
            elif path == "/recommendations":
                payload = json.loads(request.data)
                if payload.get("n") == 0 or payload["favorite_movie_ids"] == [True]:
                    status = 422
                    body = {"detail": "Invalid request"}
                else:
                    body = {
                        "method": "als"
                        if payload["favorite_movie_ids"]
                        else "popularity",
                        "model": identity,
                        "recommendations": [{"tmdb_id": 200, "score": 0.5}],
                    }
            else:
                raise AssertionError(f"Unexpected route: {path}")
            if status >= 400:
                raise HTTPError(
                    request.full_url,
                    status,
                    "Invalid request",
                    None,
                    io.BytesIO(json.dumps(body).encode()),
                )
            response = io.StringIO(json.dumps(body))
            response.status = status
            return response

        patcher = patch("scripts.check_api.urlopen", side_effect=respond)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_expected_run_is_served(self):
        self.mock_http("a" * 32)
        with patch("builtins.print"):
            check("http://api.test", [100], 1, expected_run_id="a" * 32)

    def test_other_run_is_rejected_before_recommendations(self):
        http = self.mock_http("b" * 32)
        with self.assertRaisesRegex(RuntimeError, "Unexpected model run_id"):
            check("http://api.test", [100], 1, expected_run_id="a" * 32)
        paths = [call.args[0].full_url for call in http.call_args_list]
        self.assertNotIn("http://api.test/recommendations", paths)

    def test_missing_run_is_rejected_when_expected(self):
        self.mock_http(None)
        with self.assertRaisesRegex(RuntimeError, "Unexpected model run_id"):
            check("http://api.test", [100], 1, expected_run_id="a" * 32)

    def test_synthetic_model_does_not_need_a_run_id(self):
        self.mock_http(None)
        with patch("builtins.print"):
            check("http://api.test", [100], 1)

    def test_cli_rejects_an_unexpected_run(self):
        self.mock_http("b" * 32)
        arguments = [
            "check_api.py",
            "--url",
            "http://api.test",
            "--expected-run-id",
            "a" * 32,
        ]
        with (
            patch("sys.argv", arguments),
            self.assertRaisesRegex(RuntimeError, "Unexpected model run_id"),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
