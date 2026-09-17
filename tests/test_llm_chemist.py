import unittest

from app.services.llm_chemist import clean_json_text, response_text


class LLMResponseTests(unittest.TestCase):
    def test_plain_text_response(self):
        self.assertEqual(response_text('{"ok": true}'), '{"ok": true}')

    def test_content_block_response(self):
        content = [
            {"type": "text", "text": "```json\n"},
            {"type": "text", "text": '{"ok": true}'},
            {"type": "text", "text": "\n```"},
        ]
        self.assertEqual(clean_json_text(content), '{"ok": true}')


if __name__ == "__main__":
    unittest.main()
