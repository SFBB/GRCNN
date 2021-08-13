import os
def run_tDCF(randID):
    results = os.listdir("results")
    result_ = None
    for result in results:
        if "LA_results_{}".format(randID) in result:
            result_ = result
    if result_ != None:
        os.system('python tDCF/tDCF_python_v1/evaluate_tDCF_asvspoof19.py %s %s' % ('results/{}/dev_tDCF.txt'.format(result_), '/media/ssd512/gavin/antiSpoofMusic/tDCF/tDCF_python_v1/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'))

        print('-' * 20)

        os.system('python tDCF/tDCF_python_v1/evaluate_tDCF_asvspoof19.py %s %s' % ('results/{}/eval_tDCF.txt'.format(result_), '/media/ssd512/gavin/antiSpoofMusic/tDCF/tDCF_python_v1/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'))

if __name__ == "__main__":
    run_tDCF(99999)