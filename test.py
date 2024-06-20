from utils import format_document_info_markdown


documents = [
    "The control of body weight and of blood glucose concentrations depends on the exquisite coordination of the function of several organs and tissues, in particular the liver, muscle and fat. These organs and tissues have major roles in the use and storage of nutrients in the form of glycogen or triglycerides and in the release of glucose or free fatty acids into the blood, in periods of metabolic needs. These mechanisms are tightly regulated by hormonal and nervous signals, which are generated by specialized cells that detect variations in blood glucose or lipid concentrations. The hormones insulin and glucagon not only regulate glycemic levels through their action on these organs and the sympathetic and parasympathetic branches of the autonomic nervous system, which are activated by glucose or lipid sensors, but also modulate pancreatic hormone secretion and liver, muscle and fat glucose and lipid metabolism. Other signaling molecules, such as the adipocyte hormones leptin and adiponectin, have circulating plasma concentrations that reflect the level of fat stored in adipocytes. These signals are integrated at the level of the hypothalamus by the melanocortin pathway, which produces orexigenic and anorexigenic neuropeptides to control feeding behavior, energy expenditure and glucose homeostasis.",
    "Thus, it is likely that both GLUT 2 and GK determine the set point for glucose-stimulated insulin secretion. Elucidation of distal effectors that regulate insulin secretion is also crucial to our understanding of Beta-cell function.(ABSTRACT TRUNCATED AT 250 WORDS)\n PUBMIDID(1478377) LOE(3) AUTHORS(Steiner D F,James D E)",
    "The liveror stimulates the production of glucose using these processes.In glycogenolysis, glucagon instructs the liver to convert glycogen to glucose, making glucose more available in the bloodstream.In gluconeogenesis, the liver produces glucose from the byproducts of other processes. Gluconeogenesis also occurs in the kidneys and some other organs.When the body’s glucose levels rise, insulin enables the glucose to move into cells.Insulin and glucagon work in a cycle. Glucagon interacts with the liver to increase blood sugar, while insulin reduces blood sugar by helping the cells use glucose.Blood sugar levelsA person’s blood sugar levels vary throughout the day, but insulin and glucagon keep them within a healthy range overall.When the body does not absorb or convert enough glucose, blood sugar levels remain high. Insulin reduces the body’s blood sugar levels and provides cells with glucose for energy by helping cells absorb glucose.When blood sugar levels are too low, the pancreas releases glucagon. Glucagon instructs the liver to release stored glucose, which causes the body’s blood sugar levels to rise.Hyperglycemiarefers to high blood sugar levels. Persistently high levels can cause long-term damage throughout the body.Hypoglycemiameans blood sugar levels are low.",
    "PubMed][Google Scholar]48.Poretsky L, Kalin MF. The gonadotropic function of insulin.Endocr Rev.1987;8:132–41.[PubMed][Google Scholar]49.Samoto T, Maruo T, Katayama K, Barnea ER, Mochizuk M. Altered expression of insulin and insulin-like growth factor-1 receptors in follicular and stromal compartments of polycystic ovaries.Endocr J.1993;40:413–24.[PubMed][Google Scholar]50.Abele V, Pelletier G, Tremblay RR. Radioautographic localization and regulation of the insulin receptors in rat testis.J Recept Res.1986;6:461–73.[PubMed][Google Scholar]51.Thomas DM, Udagawa N, Hards DK, et al. Insulin receptor expression in primary and cultured oseoclast-like cells.Bone.1998;23:181–6.[PubMed][Google Scholar]52.Lichtenstein AH, Schwab US. Relationship of dietary fat to glucose metabolism.Atherosclerosis.2000;150:227–43.[PubMed][Google Scholar]53.Bray GA, Lovejoy JC, Smith SR, et al. The Influence of Different Fats and Fatty Acids on Obesity, Insulin Resistance and Inflammation.J Nutr.2002;132:2488–91.[PubMed][Google Scholar]54.Sampath H, Ntambi JM. Polyunsaturated fatty acid regulation of gene expression.Nutr Rev.2004;62:333–9.[PubMed][Google Scholar]55.Simopoulos AP. Essential fatty acids in health and chronic disease.Am J Clin Nutr.1999;70(3 Suppl):560S–569S.[PubMed][Google Scholar]56.Borkman M, Chisholm DJ, Furler SM, et al. Effects of fish oil supplementation on glucose and lipid metabolism in NIDDM.Diabetes.1989;38:1314–9.[PubMed][Google Scholar]57.Friedberg CE, Janssen MJ, Heine RJ, Grobbee DE. Fish oil and glycemic control in diabetes.",
    "International Textbook of Diabetes Mellitus (2ed) John Wiley & Sons, New York; 1997 p. 469–88.31.Clemmons DR. Structural and functional analysis of insulin-like growth factors.Br Med Bull.1989;45:465–80.[PubMed][Google Scholar]32.Frystyk J, Ørskof H. IGFI, IGFII, IGF-binding proteins and diabetes. In: Alberti KGMM, Zimmet P, Defronzo RA, Keen H (hon), editors. International Textbook of Diabetes Mellitus (2ed) John Wiley & Sons, New York; 1997 p. 417–36.33.Wheatcroft SB, Williams IL, Shah AM, Kearney MT. Pathophysiological implications of insulin resistance on vascular endothelial function.Diabet Med.2003;20:255–68.[PubMed][Google Scholar]34.Smith U. Impaired (‘diabetic’) insulin signaling and action occur in fat cells long before glucose intolerance--is insulin resistance initiated in the adipose tissue?Int J Obes Relat Metab Disord.2002;26:897–904.[PubMed][Google Scholar]35.Giorgino F, Laviola L, Eriksson JW. Regional differences of insulin action in adipose tissue: insights from in vivo and in vitro studies.Acta Physiol Scand.2005;183:13–30.[PubMed][Google Scholar]36.Halvatsiotis PG, Turk D, Alzaid A, Dinneen S, Rizza RA, Nair KS. Insulin effect on leucine kinetics in type 2 diabetes mellitus.Diabetes Nutr Metab.2002;15:136–42.[PubMed][Google Scholar]37.Grundy SM. What is the contribution of obesity to the metabolic syndrome?Endocrinol Metab Clin North Am.2004;33:267–82.[PubMed][Google Scholar]38.Krauss RM, Siri PW. Metabolic abnormalities: triglyceride and low-density lipoprotein.Endocrinol Metab Clin North Am.2004;33:405–15.[PubMed][Google Scholar]39.",
    "More recently, with Alan Saghatelian's lab, we discovered a novel class of lipids with antidiabetes and anti-inflammatory effects. We also investigated the effects of the adipose-secreted hormone, leptin, on insulin sensitivity. We found that the AMP-activated protein kinase (AMPK) pathway mediates leptin's effects on fatty acid oxidation in muscle and also plays a role in leptin's anorexigenic effects in the hypothalamus. These studies transformed AMPK from a \"fuel gauge\" that regulates energy supply at the cellular level to a sensing and signaling pathway that regulates organismal energy balance. Overall, these studies have expanded our understanding of the multifaceted role of adipose tissue in metabolic health and how adipose dysfunction increases the risk for type 2 diabetes. \n PUBMIDID(30573674) LOE(4) AUTHORS(Kahn Barbara B)",
    "Insulin regulates circulating glucose levels by suppressing hepatic glucose production and increasing glucose transport into muscle and adipose tissues. Defects in these processes are associated with elevated vascular glucose levels and can lead to increased risk for the development of Type 2 diabetes mellitus and its associated disease complications. At the cellular level, insulin stimulates glucose uptake by inducing the translocation of the glucose transporter 4 (GLUT4) from intracellular storage sites to the plasma membrane, where the transporter facilitates the diffusion of glucose into striated muscle and adipocytes. Although the immediate downstream molecules that function proximal to the activated insulin receptor have been relatively well-characterized, it remains unknown how the distal insulin-signaling cascade interfaces with and recruits GLUT4 to the cell surface. New biochemical assays and imaging techniques, however, have focused attention on the plasma membrane as a potential target of insulin action leading to GLUT4 translocation. Indeed, it now appears that insulin specifically regulates the docking and/or fusion of GLUT4-vesicles with the plasma membrane. Future work will focus on identifying the key insulin targets that regulate the GLUT4 docking/fusion processes.\n PUBMIDID(17629673) LOE(6) AUTHORS(Watson Robert T,Pessin Jeffrey E)"
]



print(format_document_info_markdown("", documents))