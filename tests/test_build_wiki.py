import unittest

from reference.build_wiki import (
    _UNSORTED_KEY,
    _group_by_recueil,
    _recueil_key,
    _slugify,
    _story_slug,
)


class TestRecueilKey(unittest.TestCase):
    def test_strips_split_suffix(self):
        # "La rue broca#1".."#12" doivent se regrouper sous la même clé —
        # sans le strip, split_stories.py casse le regroupement en 12 clés
        # distinctes pour un seul recueil physique.
        self.assertEqual(_recueil_key("La rue broca#1/track1.mp3"), "La rue broca")
        self.assertEqual(_recueil_key("La rue broca#12/track1.mp3"), "La rue broca")

    def test_specific_folder_kept_as_is(self):
        self.assertEqual(_recueil_key("Agnès Chaumié/Tourneboule"), "Agnès Chaumié")

    def test_generic_bucket_goes_unsorted(self):
        self.assertEqual(_recueil_key("Contes/ma boite à histoire"), _UNSORTED_KEY)
        self.assertEqual(_recueil_key("Interprète inconnu/Winona"), _UNSORTED_KEY)

    def test_single_level_path_without_subfolder(self):
        self.assertEqual(_recueil_key("Le_BGG"), "Le_BGG")


class TestGroupByRecueil(unittest.TestCase):
    def _story(self, id_, folder_path):
        return {"id": id_, "folder_path": folder_path, "title": f"story-{id_}"}

    def test_split_siblings_regrouped_together(self):
        stories = [
            self._story(1, "La rue broca#1/t.mp3"),
            self._story(2, "La rue broca#2/t.mp3"),
            self._story(3, "La rue broca#3/t.mp3"),
        ]
        groups = _group_by_recueil(stories)
        self.assertIn("La rue broca", groups)
        self.assertEqual({s["id"] for s in groups["La rue broca"]}, {1, 2, 3})
        # _UNSORTED_KEY est toujours présent (repli initialisé même vide),
        # mais ne doit contenir aucune de ces 3 histoires regroupées.
        self.assertEqual(groups[_UNSORTED_KEY], [])

    def test_singleton_specific_folder_falls_back_to_unsorted(self):
        # Un dossier spécifique (pas générique) mais à une seule histoire ne
        # justifie pas sa propre page de recueil.
        stories = [self._story(1, "Antoine de Saint-Exupéry/Le petit prince")]
        groups = _group_by_recueil(stories)
        self.assertEqual(groups[_UNSORTED_KEY], stories)
        self.assertNotIn("Antoine de Saint-Exupéry", groups)

    def test_generic_bucket_always_unsorted_even_with_many_members(self):
        stories = [self._story(i, f"Contes/histoire-{i}") for i in range(5)]
        groups = _group_by_recueil(stories)
        self.assertEqual({s["id"] for s in groups[_UNSORTED_KEY]}, {0, 1, 2, 3, 4})

    def test_real_multi_story_collection_kept_as_group(self):
        stories = [self._story(i, f"Philippe Lejour/conte-{i}") for i in range(3)]
        groups = _group_by_recueil(stories)
        self.assertEqual(len(groups["Philippe Lejour"]), 3)


class TestSlug(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(_story_slug("Le Petit Prince", 42), _story_slug("Le Petit Prince", 42))

    def test_accents_and_case_normalized(self):
        self.assertEqual(_slugify("Agnès Chaumié"), "agnes-chaumie")

    def test_story_id_suffix_avoids_collision_between_same_titles(self):
        self.assertNotEqual(_story_slug("Sans titre", 1), _story_slug("Sans titre", 2))

    def test_empty_title_falls_back_to_placeholder(self):
        self.assertTrue(_slugify("###").startswith("x"))


if __name__ == "__main__":
    unittest.main()
