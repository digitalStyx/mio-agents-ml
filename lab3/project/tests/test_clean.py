"""Krok 2 planu: czyszczenie na krótkich przykładach."""
import unittest

from src.clean import clean
from src.data import load_config
from src.features import make_analyzer

ILUG = ("Subject: [ILUG] Cheap toner cartridges\n\n"
        "Buy now and save money. Click below to remove yourself.\n\n"
        "-- \n"
        "Irish Linux Users' Group: ilug@linux.ie\n"
        "http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information.\n"
        "List maintainer: listmaster@linux.ie\n")

MAILMAN = ("Subject: Re: [Spambayes] tokenizer question\n\n"
           "On Tue, 10 Sep 2002, Tim Peters wrote:\n"
           "> I tried the new tokenizer\n"
           "> and it works.\n\n"
           "Great, thanks.\n\n"
           "_______________________________________________\n"
           "Spambayes mailing list\n"
           "Spambayes@python.org\n"
           "http://mail.python.org/mailman/listinfo/spambayes\n")

ANON = ("Subject: Account zzzz@spamassassin.taint.org\n\n"
        "User Name: zzzz@spamassassin.taint.org\n"
        "Unsubscribe: http://www.example.com/trim/?zzzz@spamassassin.taint.org%7C17\n")


class TestClean(unittest.TestCase):
    def test_list_footer_removed(self):
        for text, traces in ((ILUG, ["listinfo", "listmaster", "Irish Linux", "mailman"]),
                             (MAILMAN, ["listinfo", "mailing list", "mailman", "Spambayes@python.org"])):
            _, body = clean(text)
            for t in traces:
                self.assertNotIn(t, body)

    def test_real_content_kept(self):
        _, body = clean(ILUG)
        self.assertIn("Buy now and save money. Click below to remove yourself.", body)

    def test_subject_list_tag_removed_re_kept(self):
        self.assertEqual(clean(ILUG)[0], "Cheap toner cartridges")
        self.assertEqual(clean(MAILMAN)[0], "Re: tokenizer question")
        self.assertEqual(clean("Subject: Re: [zzzzteana] That wacky imam\n\nx")[0], "Re: That wacky imam")

    def test_anonymized_address_becomes_token(self):
        subject, body = clean(ANON, anon_token="anonaddr")
        both = subject + "\n" + body
        self.assertNotIn("zzzz", both)
        self.assertNotIn("taint", both)
        self.assertEqual(both.count("anonaddr"), 3)   # temat, treść i link

    def test_quotes_and_wrote_kept(self):
        _, body = clean(MAILMAN)
        self.assertIn("Tim Peters wrote:", body)
        self.assertIn("> I tried the new tokenizer", body)
        self.assertIn("> and it works.", body)

    def test_analyzer_marks_subject_words(self):
        toks = make_analyzer(load_config())(MAILMAN)
        self.assertIn("subj_re", toks)
        self.assertIn("subj_tokenizer", toks)
        self.assertNotIn("subj_spambayes", toks)    # znacznik listy usunięty z tematu
        self.assertIn("wrote", toks)
        self.assertNotIn("listinfo", toks)


if __name__ == "__main__":
    unittest.main()
