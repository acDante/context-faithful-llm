from prefs.atomic_facts import AtomicFactGenerator
from prefs.factscorer import FactScorer


def filter_facts(facts):
    """Some facts are vacuous of any real information, exclude these before scoring."""
    bad_substrings = ['someone','something','somebody','is a person','is a character', 'are people', 'are characters']
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or len(f.split())>2]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or (not any(x in f.lower() for x in bad_substrings))]
    # facts = [f for f in facts if not f.lower().startswith('there is a') and not 'is in a room' in f and not 'is talking' in f and not 'are talking' in f and not 'made a statement' in f]
    facts = [f for f in facts if not 'is mentioned' in f.lower() and not 'are mentioned' in f.lower() and not 'is there' in f.lower() and not 'are there' in f.lower()]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or not f.endswith(' to')]
    facts = [f for f in facts if not 'is there' in f and not 'are there' in f]
    return facts

# Evaluate on XSum test data
example_output = "Prison Link Cymru, a charity that helps ex-offenders find accommodation, claims that investing in housing for former prisoners would be cheaper than jailing them, as many struggle to find suitable housing, with some living rough for up to a year, and that more investment in one-bedroom flats could help ease the problem."
gold_summary = "There is a \"chronic\" need for more housing for prison leavers in Wales, according to a charity."
# example_output = 'Jim is the man who works at the shop. Bob also works at the shop. Mick has a cat.'
# gold_summary = 'Jim and Bob work at the shop.'

afg = AtomicFactGenerator()
predicted_facts_and_sources = afg.extract_facts(example_output) # list of (sent, facts) tuples
predicted_facts = [x for item in predicted_facts_and_sources for x in item[1]]
predicted_facts = filter_facts(predicted_facts)
print("After filtering: ", predicted_facts)

gold_facts_and_sources = afg.extract_facts(gold_summary) # list of (sent, facts) tuples
gold_facts = [x for item in gold_facts_and_sources for x in item[1]]
gold_facts = filter_facts(gold_facts)
print("After filtering: ", gold_facts)

fs = FactScorer(cache_dir_prefix='.')
fact_precision, score_per_fact = fs.get_score(predicted_facts,
                                              gold_summary,
                                              summname='test-summary',
                                              )

fact_recall, score_per_fact = fs.get_score(gold_facts,
                                           example_output,
                                           summname='test-summary',
                                           )

prefs_score = (2 * fact_precision*fact_recall) / (fact_precision + fact_recall)

# Print PREFS scores
print("fact_precision: ", fact_precision)
print("fact_recall: ", fact_recall)
print("prefs_score: ", prefs_score)