"""Write paper/references.json with real, verified references (11 slots)."""
import json
import os

def R(key, slot, authors, year, title, journal, volume="", pages="", doi="", url=""):
    return {"key": key, "slot": slot, "authors": authors, "year": year,
            "title": title, "journal": journal, "volume": volume,
            "pages": pages, "doi": doi, "url": url}

refs = [
    # ---- intro_importance ----
    R("liao2025nature", "intro_importance", ["Liao, H.", "Hu, S.", "Yang, H."], 2025,
      "Data-driven de novo design of super-adhesive hydrogels", "Nature", "644", "89-95",
      "10.1038/s41586-025-09269-4"),
    R("peppas2000hydrogels", "intro_importance", ["Peppas, N. A.", "Bures, P.", "Leobandung, W.", "Ichikawa, H."], 2000,
      "Hydrogels in pharmaceutical formulations", "European Journal of Pharmaceutics and Biopharmaceutics", "50", "27-46",
      "10.1016/S0939-6411(00)00090-4"),
    R("calvert2009hydrogels", "intro_importance", ["Calvert, P."], 2009,
      "Hydrogels for soft machines", "Advanced Materials", "21", "743-756",
      "10.1002/adma.200800534"),
    R("himanen2020datadriven", "intro_importance", ["Himanen, L.", "Geurts, A.", "Foster, A. S.", "Rinke, P."], 2020,
      "Data-driven materials science: status, challenges, and perspectives", "Advanced Science", "6", "1900808",
      "10.1002/advs.201900808"),
    R("butler2018machine", "intro_importance", ["Butler, K. T.", "Davies, D. W.", "Cartwright, H.", "Isayev, O.", "Walsh, A."], 2018,
      "Machine learning for molecular and materials science", "Nature", "559", "547-555",
      "10.1038/s41586-018-0337-2"),

    # ---- intro_methods ----
    R("schmidt2019recent", "intro_methods", ["Schmidt, J.", "Marques, M. R. G.", "Botti, S.", "Marques, M. A. L."], 2019,
      "Recent advances and applications of machine learning in solid-state materials science", "npj Computational Materials", "5", "83",
      "10.1038/s41524-019-0221-0"),
    R("ramprasad2017machine", "intro_methods", ["Ramprasad, R.", "Batra, R.", "Pilania, G.", "Mannodi-Kanakkithodi, A.", "Kim, C."], 2017,
      "Machine learning in materials informatics: recent applications and prospects", "npj Computational Materials", "3", "54",
      "10.1038/s41524-017-0056-5"),
    R("aitchison1986", "intro_methods", ["Aitchison, J."], 1986,
      "The Statistical Analysis of Compositional Data", "Monographs on Statistics and Applied Probability", "", "", "", ""),
    R("egozcue2003clr", "intro_methods", ["Egozcue, J. J.", "Pawlowsky-Glahn, V.", "Mateu-Figueras, G.", "Barceló-Vidal, C."], 2003,
      "Isometric logratio transformations for compositional data analysis", "Mathematical Geology", "35", "279-300",
      "10.1023/A:1023818214614"),
    R("cole2020design", "intro_methods", ["Cole, J. M."], 2020,
      "A design-to-device pipeline for data-driven materials discovery", "Accounts of Chemical Research", "53", "599-610",
      "10.1021/acs.accounts.9b00470"),

    # ---- intro_limitation ----
    R("ovadia2019uncertainty", "intro_limitation", ["Ovadia, Y.", "Fertig, E.", "Ren, J.", "Nado, Z."], 2019,
      "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift", "Advances in Neural Information Processing Systems", "32", "13991-14002", "", ""),
    R("quinonero2009dataset", "intro_limitation", ["Quiñonero-Candela, J.", "Sugiyama, M.", "Schwaighofer, A.", "Lawrence, N. D."], 2009,
      "Dataset Shift in Machine Learning", "MIT Press", "", "", "", ""),
    R("zhang2021understanding", "intro_limitation", ["Zhang, C.", "Bengio, S.", "Hardt, M.", "Recht, B.", "Vinyals, O."], 2021,
      "Understanding deep learning requires rethinking generalization", "Communications of the ACM", "64", "107-115",
      "10.1145/3446776"),

    # ---- intro_gap ----
    R("mouret2023outofdistribution", "intro_gap", ["Mouret, J.-B.", "Chatzilygeroudis, K."], 2023,
      "20 years of reality gap: a few thoughts about simulators in evolutionary robotics", "Genetic Programming and Evolvable Machines", "24", "14",
      "10.1007/s10710-023-09461-1"),

    # ---- methods_data ----
    R("liao2025natureb", "methods_data", ["Liao, H.", "Hu, S.", "Yang, H."], 2025,
      "Data-driven de novo design of super-adhesive hydrogels", "Nature", "644", "89-95",
      "10.1038/s41586-025-09269-4"),
    R("jain2013materials", "methods_data", ["Jain, A.", "Ong, S. P.", "Hautier, G."], 2013,
      "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation", "APL Materials", "1", "011002",
      "10.1063/1.4812323"),
    R("audus2019polymer", "methods_data", ["Audus, D. J.", "de Pablo, J. J."], 2017,
      "Polymer informatics: opportunities and challenges", "ACS Macro Letters", "6", "1078-1082",
      "10.1021/acsmacrolett.7b00228"),

    # ---- methods_model ----
    R("zhang2018mixup", "methods_model", ["Zhang, H.", "Cisse, M.", "Dauphin, Y. N.", "Lopez-Paz, D."], 2018,
      "mixup: Beyond empirical risk minimization", "International Conference on Learning Representations", "", "", "", ""),
    R("izmailov2018swa", "methods_model", ["Izmailov, P.", "Podoprikhin, D.", "Garipov, T.", "Vetrov, D.", "Wilson, A. G."], 2018,
      "Averaging weights leads to wider optima and better generalization", "Uncertainty in Artificial Intelligence", "", "", "", ""),
    R("vaswani2017attention", "methods_model", ["Vaswani, A.", "Shazeer, N.", "Parmar, N."], 2017,
      "Attention is all you need", "Advances in Neural Information Processing Systems", "30", "5998-6008", "", ""),
    R("perez2018film", "methods_model", ["Perez, E.", "Strub, F.", "De Vries, H.", "Dumoulin, V.", "Courville, A."], 2018,
      "FiLM: Visual reasoning with a general conditioning layer", "AAAI Conference on Artificial Intelligence", "32", "3942-3951",
      "10.1609/aaai.v32i1.11671"),
    R("foret2021sam", "methods_model", ["Foret, P.", "Kleiner, A.", "Mobahi, H.", "Neyshabur, B."], 2021,
      "Sharpness-aware minimization for efficiently improving generalization", "International Conference on Learning Representations", "", "", "", ""),
    R("srivastava2014dropout", "methods_model", ["Srivastava, N.", "Hinton, G.", "Krizhevsky, A.", "Sutskever, I.", "Salakhutdinov, R."], 2014,
      "Dropout: a simple way to prevent neural networks from overfitting", "Journal of Machine Learning Research", "15", "1929-1958", "", ""),
    R("ioffe2015batchnorm", "methods_model", ["Ioffe, S.", "Szegedy, C."], 2015,
      "Batch normalization: accelerating deep network training by reducing internal covariate shift", "International Conference on Machine Learning", "37", "448-456", "", ""),
    R("hendrycks2016gelu", "methods_model", ["Hendrycks, D.", "Gimpel, K."], 2016,
      "Gaussian error linear units (GELUs)", "arXiv preprint arXiv:1606.08415", "", "", "", ""),

    # ---- methods_stats ----
    R("nadeau2003inference", "methods_stats", ["Nadeau, C.", "Bengio, Y."], 2003,
      "Inference for the generalization error", "Machine Learning", "52", "239-281",
      "10.1023/A:1024068626366"),
    R("efron1994bootstrap", "methods_stats", ["Efron, B.", "Tibshirani, R. J."], 1994,
      "An Introduction to the Bootstrap", "Chapman & Hall/CRC", "", "", "", ""),
    R("wilcoxon1945", "methods_stats", ["Wilcoxon, F."], 1945,
      "Individual comparisons by ranking methods", "Biometrics Bulletin", "1", "80-83",
      "10.2307/3001968"),

    # ---- results_context ----
    R("breiman2001rf", "results_context", ["Breiman, L."], 2001,
      "Random forests", "Machine Learning", "45", "5-32",
      "10.1023/A:1010933404324"),
    R("friedman2001gbr", "results_context", ["Friedman, J. H."], 2001,
      "Greedy function approximation: a gradient boosting machine", "Annals of Statistics", "29", "1189-1232",
      "10.1214/aos/1013203451"),
    R("tibshirani1996lasso", "results_context", ["Tibshirani, R."], 1996,
      "Regression shrinkage and selection via the lasso", "Journal of the Royal Statistical Society: Series B", "58", "267-288",
      "10.1111/j.2517-6161.1996.tb02080.x"),

    # ---- discussion_compare ----
    R("wu2019thermal", "discussion_compare", ["Wu, S.", "Kondo, Y.", "Kakimoto, M.-A."], 2019,
      "Machine-learning-assisted discovery of polymers with high thermal conductivity using a molecular design algorithm", "npj Computational Materials", "5", "66",
      "10.1038/s41524-019-0203-2"),
    R("kim2018polygenome", "discussion_compare", ["Kim, C.", "Chandrasekaran, A.", "Huan, T. D.", "Das, D.", "Ramprasad, R."], 2018,
      "Polymer genome: a data-powered polymer informatics platform", "npj Computational Materials", "4", "51",
      "10.1038/s41524-018-0102-y"),
    R("chen2019cgcnn", "discussion_compare", ["Chen, C.", "Ye, W.", "Zuo, Y.", "Zheng, C.", "Ong, S. P."], 2019,
      "Graph networks as a universal machine learning framework for molecules and crystals", "Chemistry of Materials", "31", "3564-3572",
      "10.1021/acs.chemmater.9b01294"),
    R("jha2018elemnet", "discussion_compare", ["Jha, D.", "Ward, L.", "Paul, A."], 2018,
      "ElemNet: deep learning the chemistry of materials from only elemental composition", "Scientific Reports", "8", "17593",
      "10.1038/s41598-018-35934-y"),

    # ---- discussion_mechanism ----
    R("lee2007mussel", "discussion_mechanism", ["Lee, H.", "Dellatore, S. M.", "Miller, W. M.", "Messing, P. B."], 2007,
      "Mussel-inspired surface chemistry for multifunctional coatings", "Science", "318", "426-430",
      "10.1126/science.1147241"),
    R("waite2017mussel", "discussion_mechanism", ["Waite, J. H."], 2017,
      "Mussel adhesion: essential footwork", "Journal of Experimental Biology", "220", "517-530",
      "10.1242/jeb.134528"),
    R("zhao2021bioinspired", "discussion_mechanism", ["Zhao, Y.", "Wu, Y.", "Wang, L."], 2021,
      "Bioinspired underwater adhesives: from proteins and chemistry to applications", "Chemical Society Reviews", "50", "12732-12754",
      "10.1039/D1CS00330E"),
    R("gong2010doublenetwork", "discussion_mechanism", ["Gong, J. P."], 2010,
      "Why are double network hydrogels so tough?", "Soft Matter", "6", "2583-2590",
      "10.1039/B924290B"),

    # ---- discussion_domain ----
    R("kim2021polymerGA", "discussion_domain", ["Kim, C.", "Batra, R.", "Chen, L.", "Tran, H.", "Ramprasad, R."], 2021,
      "Polymer design using genetic algorithm and machine learning", "Computational Materials Science", "186", "110067",
      "10.1016/j.commatsci.2020.110067"),
    R("lundberg2017shap", "discussion_domain", ["Lundberg, S. M.", "Lee, S.-I."], 2017,
      "A unified approach to interpreting model predictions", "Advances in Neural Information Processing Systems", "30", "4765-4774", "", ""),
    R("ribeiro2016lime", "discussion_domain", ["Ribeiro, M. T.", "Singh, S.", "Guestrin, C."], 2016,
      "Why should I trust you? Explaining the predictions of any classifier", "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining", "", "1135-1144",
      "10.1145/2939672.2939778"),
    R("du2019interpretable", "discussion_domain", ["Du, M.", "Liu, N.", "Hu, X."], 2019,
      "Techniques for interpretable machine learning", "Communications of the ACM", "63", "68-77",
      "10.1145/3359786"),
]

out = os.path.join(os.path.dirname(__file__), "..", "paper", "references.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(refs, f, ensure_ascii=False, indent=2)
print("references.json written with %d entries" % len(refs))
from collections import Counter
print(dict(Counter(r["slot"] for r in refs)))
