#!/usr/bin/env python3
"""
French localisation for the participant report.

The renderer emits English HTML; `translate(html)` rewrites the participant-facing
text to French in a single post-render pass. Because every user-facing string ends
up in the final HTML, one ordered phrase map covers both helper-generated and inline
strings. Replacements are applied longest-key-first so long phrases are handled
before any shorter substring of them (e.g. "FEV1 / FVC" before "FVC").

Units left untranslated on purpose: bpm, ms, s, mmHg, L, mL, %, cm, cm/s, kg, %/mmHg.
Reference citations (author names / journals) are left as-is.

If a source string changes, its French mapping falls back to English — run
`--lang fr` and scan for leftover English to catch drift.
"""

# English -> French. Keep keys EXACTLY as they appear in the rendered HTML.
FR = {
    # ---- document / language ------------------------------------------------
    '<html lang="en">': '<html lang="fr">',
    'LC Study Participant Testing Report': 'Rapport de tests du participant — Étude LC',
    'Research use only': 'Usage de recherche uniquement',
    'LC Study': 'Étude LC',

    # ---- overview page ------------------------------------------------------
    'Physiological &amp; neuropsychological testing': 'Tests physiologiques et neuropsychologiques',
    'Physiological Testing Report': 'Rapport de tests physiologiques',
    '01 / Overview': '01 / Aperçu',
    '<span>Participant</span>': '<span>Participant</span>',
    '<span>Session</span>': '<span>Séance</span>',
    '<span>Age</span>': '<span>Âge</span>',
    ' years</strong>': ' ans</strong>',
    '<span>Scan time</span>': '<span>Heure de l’examen</span>',
    '<span>Height</span>': '<span>Taille</span>',
    '<span>Weight</span>': '<span>Poids</span>',
    'Report map': 'Plan du rapport',
    'Not for clinical use': 'Pas à usage clinique',
    'This report summarizes research assessments and does not carry diagnostic or prescriptive authority. Measurements can vary with your physiological state on the day of testing.':
        'Ce rapport résume des évaluations de recherche et n’a aucune valeur diagnostique ou de prescription. Les mesures peuvent varier selon votre état physiologique le jour du test.',
    'Incomplete source data': 'Données sources incomplètes',
    'No neuropsychological assessment record was found for this subject in the source dataset.':
        'Aucun dossier d’évaluation neuropsychologique n’a été trouvé pour ce participant dans les données sources.',
    'Metadata fallback': 'Métadonnées de secours',

    # ---- chapter-map rows ---------------------------------------------------
    'MoCA score and reaction indices': 'Score MoCA et indices de réaction',
    'FVC, FEV1, ratio, and peak flow': 'FVC, FEV1, rapport et débit de pointe',
    'Resting cardiovascular': 'Cardiovasculaire au repos',
    'Heart rate variability and blood pressure': 'Variabilité de la fréquence cardiaque et pression artérielle',
    'Figures added': 'Figures ajoutées',
    'Supine-to-stand, Valsalva, and deep breathing': 'Coucher-debout, Valsalva et respiration profonde',
    'Grouped chapter': 'Chapitre groupé',
    'Gas manipulation': 'Manipulation des gaz',
    'CO₂/O₂ end-tidal, SpO₂, and Doppler velocity': 'CO₂/O₂ télé-expiratoires, SpO₂ et vélocité Doppler',
    'Separate page': 'Page distincte',

    # ---- cognitive page -----------------------------------------------------
    'Section 01': 'Section 01',
    'Cognitive testing': 'Tests cognitifs',
    '02 / Cognitive': '02 / Cognition',
    'MoCA-Blind total (raw)': 'Total MoCA-Aveugle (brut)',
    'Participant context': 'Contexte du participant',
    'Raw MoCA-Blind screening score from the source dataset.':
        'Score brut de dépistage MoCA-Aveugle issu des données sources.',
    'This subject has no neuropsychological assessment on record in the source dataset.':
        'Ce participant n’a aucune évaluation neuropsychologique enregistrée dans les données sources.',
    'A neuropsych record exists for this subject, but the MoCA-Blind total is not filled in yet.':
        'Un dossier neuropsychologique existe pour ce participant, mais le total MoCA-Aveugle n’est pas encore renseigné.',
    'The MoCA-Blind is scored out of 22 (the eight sight-dependent points are not administered). '
    'A raw MoCA-Blind score below 18/22 may be indicative of mild cognitive impairment. This is a '
    'screening measure, not a diagnosis — a single score is one point-in-time research measurement, '
    'and performance may be influenced by language, education, fatigue, hearing, and the testing '
    'environment. A healthcare professional can provide appropriate follow-up interpretation.':
        'Le MoCA-Aveugle est noté sur 22 (les huit points dépendant de la vue ne sont pas administrés). '
        'Un score brut MoCA-Aveugle inférieur à 18/22 peut évoquer un trouble cognitif léger. Il s’agit d’un '
        'outil de dépistage, non d’un diagnostic — un score isolé est une mesure de recherche à un instant donné, '
        'et la performance peut être influencée par la langue, la scolarité, la fatigue, l’audition et les '
        'conditions du test. Un professionnel de la santé peut fournir une interprétation de suivi appropriée.',
    'MoCA-Blind subscores': 'Sous-scores MoCA-Aveugle',
    'Digit span': 'Empan de chiffres',
    'Vigilance': 'Vigilance',
    'Serial 7s': 'Soustractions de 7',
    'Sentence repetition': 'Répétition de phrases',
    'Verbal fluency': 'Fluence verbale',
    'Abstraction': 'Abstraction',
    'Delayed recall': 'Rappel différé',
    'Orientation': 'Orientation',
    'About this assessment': 'À propos de cette évaluation',
    'The MoCA is a screening assessment, not a diagnosis. Performance may be influenced by language, education, fatigue, hearing, vision, and the testing environment.':
        'Le MoCA est un outil de dépistage, non un diagnostic. La performance peut être influencée par la langue, la scolarité, la fatigue, l’audition, la vision et les conditions du test.',

    # ---- spirometry page ----------------------------------------------------
    'Section 02': 'Section 02',
    'Spirometry': 'Spirométrie',
    '03 / Respiratory': '03 / Respiratoire',
    'Spirometry measures how much air you can exhale and how quickly you can exhale it.':
        'La spirométrie mesure la quantité d’air que vous pouvez expirer et la vitesse à laquelle vous l’expirez.',
    'First-second volume': 'Volume de la première seconde',
    'Forced vital capacity': 'Capacité vitale forcée',
    'Calculated from reported values': 'Calculé à partir des valeurs rapportées',
    'Volume comparison': 'Comparaison des volumes',
    'Participant values and expected reference range': 'Valeurs du participant et intervalle de référence attendu',
    'Illustrative scale': 'Échelle illustrative',
    'The shaded band represents the literature-predicted (Bowerman, 2022) values for the FVC maneuver, '
    'based on demographic information. If it is not displayed, we may be missing some demographic information '
    'from you (age, sex, or height). The vertical mark indicates where your result lies against the predicted '
    'values, though it may be impacted by the quality of the spirometry maneuvers performed.':
        'La bande ombrée représente les valeurs prédites par la littérature (Bowerman, 2022) pour la manœuvre de CVF, '
        'd’après des données démographiques. Si elle ne s’affiche pas, il se peut qu’il manque certaines de vos données '
        'démographiques (âge, sexe ou taille). Le repère vertical indique où se situe votre résultat par rapport aux '
        'valeurs prédites, bien qu’il puisse être affecté par la qualité des manœuvres de spirométrie réalisées.',
    'This is not clinical advice. If you have any questions or concerns regarding '
    'these results, please talk to a doctor.':
        'Ceci n’est pas un avis clinique. Si vous avez des questions ou des préoccupations concernant '
        'ces résultats, veuillez en parler à un médecin.',
    'How to read spirometry values': 'Comment lire les valeurs de spirométrie',
    'FEV1 is the forced expiratory volume in one second and FVC is the forced vital capacity, and FEV1/FVC is a ratio typically used for clinical diagnosis of obstructed lung disorders. Interpretation typically considers age, sex, height, reference equations, and test quality. These results should not be substituted for medical advice; please consult a doctor if you have any concerns.':
        'Le FEV1 est le volume expiratoire maximal en une seconde et la FVC est la capacité vitale forcée ; le rapport FEV1/FVC sert habituellement au diagnostic clinique des troubles obstructifs pulmonaires. L’interprétation tient généralement compte de l’âge, du sexe, de la taille, des équations de référence et de la qualité du test. Ces résultats ne remplacent pas un avis médical ; veuillez consulter un médecin en cas de préoccupation.',

    # ---- resting page -------------------------------------------------------
    'Section 03': 'Section 03',
    'Cardiovascular Function At Rest': 'Fonction cardiovasculaire au repos',
    '04 / Resting state': '04 / État de repos',
    '1. ECG measures': '1. Mesures ECG',
    '2. Blood pressure measures': '2. Mesures de pression artérielle',
    '3. Respiratory measures': '3. Mesures respiratoires',
    '4. Doppler measures': '4. Mesures Doppler',
    '<span>Mean heart rate</span>': '<span>Fréquence cardiaque moyenne</span>',
    '<span>Mean RR</span>': '<span>RR moyen</span>',
    '<span>Systolic</span>': '<span>Systolique</span>',
    '<span>Mean arterial</span>': '<span>Artérielle moyenne</span>',
    '<span>Diastolic</span>': '<span>Diastolique</span>',
    '<span>Source</span>': '<span>Source</span>',
    'Continuous ABP': 'PA continue',
    '<span>End-tidal CO2</span>': '<span>CO2 télé-expiratoire</span>',
    '<span>Breathing rate</span>': '<span>Fréquence respiratoire</span>',
    '<span>Tidal volume</span>': '<span>Volume courant</span>',
    '<span>Minute ventilation</span>': '<span>Ventilation minute</span>',
    'breaths/min': 'resp/min',
    '<span>Mean peak velocity</span>': '<span>Vélocité de pointe moyenne</span>',
    '<span>Mean trough velocity</span>': '<span>Vélocité minimale moyenne</span>',
    '<span>Mean flow velocity</span>': '<span>Vélocité de flux moyenne</span>',
    '<span>Mean quality</span>': '<span>Qualité moyenne</span>',
    '<span>LF / HF ratio</span>': '<span>Rapport LF / HF</span>',
    '>ECG / heart rate<': '>ECG / fréquence cardiaque<',
    '>Blood pressure<': '>Pression artérielle<',
    '>Respiratory<': '>Respiratoire<',
    'Average resting ECG waveform': 'Tracé ECG moyen au repos',
    '<span class="metric-label">P duration</span>': '<span class="metric-label">Durée P</span>',
    '<span class="metric-label">QRS duration</span>': '<span class="metric-label">Durée QRS</span>',
    '<span class="metric-label">PQ time</span>': '<span class="metric-label">Temps PQ</span>',
    '<span class="metric-label">QT time</span>': '<span class="metric-label">Temps QT</span>',
    'About HRV measures': 'À propos des mesures de VFC',
    'RMSSD is a time-domain heart-rate-variability measure associated primarily with parasympathetic activity. LF/HF is often reported as a frequency-domain index; interpretation remains context dependent.':
        'Le RMSSD est une mesure temporelle de la variabilité de la fréquence cardiaque associée surtout à l’activité parasympathique. Le rapport LF/HF est souvent rapporté comme indice fréquentiel ; son interprétation demeure dépendante du contexte.',

    # ---- autonomic overview -------------------------------------------------
    'Section 04': 'Section 04',
    'Autonomic testing': 'Tests autonomiques',
    '05 / Chapter overview': '05 / Aperçu du chapitre',
    'The autonomic nervous system helps regulate involuntary functions such as heart rate and blood pressure.':
        'Le système nerveux autonome aide à réguler des fonctions involontaires comme la fréquence cardiaque et la pression artérielle.',
    'One coherent chapter': 'Un chapitre cohérent',
    'Response to posture, strain, and breathing': 'Réponse à la posture, à l’effort et à la respiration',
    'Review the trends first, then the calculated indices and participant-oriented explanation.':
        'Examinez d’abord les tendances, puis les indices calculés et l’explication destinée au participant.',
    '01 · Supine-to-stand response': '01 · Réponse coucher-debout',
    '02 · Valsalva maneuver': '02 · Manœuvre de Valsalva',
    '03 · Deep breathing response': '03 · Réponse à la respiration profonde',
    'Rest quietly while lying down, then stand when instructed while heart rate and blood pressure continue recording.':
        'Reposez-vous calmement en position allongée, puis levez-vous à la consigne pendant que la fréquence cardiaque et la pression artérielle continuent d’être enregistrées.',
    'Blow steadily into a syringe against resistance for the instructed interval, followed by recovery, while synchronized signals are recorded.':
        'Soufflez de façon constante dans une seringue contre résistance pendant l’intervalle indiqué, suivi d’une récupération, pendant que des signaux synchronisés sont enregistrés.',
    'Follow paced inhale and exhale cues so breathing-linked changes in heart rate can be assessed.':
        'Suivez les consignes rythmées d’inspiration et d’expiration afin d’évaluer les variations de la fréquence cardiaque liées à la respiration.',

    # ---- supine-to-stand ----------------------------------------------------
    'Autonomic testing · 01': 'Tests autonomiques · 01',
    'Supine-to-stand response': 'Réponse coucher-debout',
    'Supine to stand': 'Coucher-debout',
    '06 / STS': '06 / CD',
    'This assessment summarizes the change in heart rate and blood pressure from lying down to standing.':
        'Cette évaluation résume la variation de la fréquence cardiaque et de la pression artérielle entre la position allongée et la position debout.',
    '<span class="metric-label">Baseline HR</span>': '<span class="metric-label">FC de base</span>',
    '<span class="metric-label">Plateau HR</span>': '<span class="metric-label">FC en plateau</span>',
    '<span class="metric-label">Delta HR</span>': '<span class="metric-label">Delta FC</span>',
    '<span class="metric-label">Delta BP</span>': '<span class="metric-label">Delta PA</span>',
    'About orthostatic response': 'À propos de la réponse orthostatique',
    'Orthostatic intolerance describes symptoms that occur on standing and improve when lying down. Formal interpretation considers symptoms, timing, heart-rate change, blood-pressure change, medications, and clinical context.':
        'L’intolérance orthostatique désigne des symptômes qui surviennent au passage debout et s’améliorent en position allongée. Une interprétation formelle tient compte des symptômes, du délai, de la variation de la fréquence cardiaque et de la pression artérielle, des médicaments et du contexte clinique.',

    # ---- valsalva -----------------------------------------------------------
    'Autonomic testing · 02': 'Tests autonomiques · 02',
    'Valsalva maneuver': 'Manœuvre de Valsalva',
    '07 / Valsalva': '07 / Valsalva',
    'The Valsalva maneuver records cardiovascular responses during a controlled strain and recovery.':
        'La manœuvre de Valsalva enregistre les réponses cardiovasculaires pendant un effort contrôlé et la récupération.',
    '<span class="metric-label">Valsalva ratio</span>': '<span class="metric-label">Rapport de Valsalva</span>',
    'Calculated from artifact-rejected median HR': 'Calculé à partir de la FC médiane après rejet des artéfacts',
    '<span class="metric-label">Late Phase II MAP change</span>': '<span class="metric-label">Variation de PAM en Phase II tardive</span>',
    'Recovery relative to early Phase II nadir': 'Récupération par rapport au nadir de la Phase II précoce',
    '<span class="metric-label">Phase IV MAP change</span>': '<span class="metric-label">Variation de PAM en Phase IV</span>',
    'Rise relative to Phase III nadir': 'Hausse par rapport au nadir de la Phase III',
    'Valsalva response': 'Réponse de Valsalva',

    # ---- deep breathing -----------------------------------------------------
    'Autonomic testing · 03': 'Tests autonomiques · 03',
    'Deep breathing': 'Respiration profonde',
    '08 / Deep breathing': '08 / Respiration profonde',
    'The deep-breathing assessment evaluates heart-rate variation across slow breathing cycles, a response associated with cardiovagal function.':
        'L’évaluation de la respiration profonde mesure la variation de la fréquence cardiaque au fil de cycles respiratoires lents, une réponse associée à la fonction cardiovagale.',
    '<span class="metric-label">E:I ratio</span>': '<span class="metric-label">Rapport E:I</span>',
    'Deep breathing heart-rate response': 'Réponse de la fréquence cardiaque à la respiration profonde',
    'About E:I ratio': 'À propos du rapport E:I',
    'The expiratory-to-inspiratory ratio compares the longest RR interval during expiration with the shortest RR interval during inspiration. It is interpreted in relation to age, test conditions, and other autonomic measures.':
        'Le rapport expiratoire/inspiratoire compare le plus long intervalle RR pendant l’expiration au plus court intervalle RR pendant l’inspiration. Il s’interprète en fonction de l’âge, des conditions du test et d’autres mesures autonomiques.',

    # ---- gas page -----------------------------------------------------------
    'Section 06': 'Section 06',
    '09 / Gas manipulation': '09 / Manipulation des gaz',
    '<span class="metric-label">Cerebrovascular reactivity</span>': '<span class="metric-label">Réactivité cérébrovasculaire</span>',
    'Doppler velocity response to CO₂ during hypercapnia': 'Réponse de la vélocité Doppler au CO₂ pendant l’hypercapnie',
    '<span class="metric-label">Minimum SpO₂</span>': '<span class="metric-label">SpO₂ minimale</span>',
    'Lowest oxygen saturation during hypoxia': 'Saturation en oxygène la plus basse pendant l’hypoxie',
    'About the gas-manipulation task': 'À propos de la tâche de manipulation des gaz',
    'Inspired gas is manipulated to raise CO₂ (hypercapnia) and lower O₂ (hypoxia) while '
    'cerebral blood-flow velocity is recorded by transcranial Doppler ultrasound and SpO₂, the '
    'oxygen saturation, is measured from a pulse-oximeter placed on the finger. '
    'End-tidal CO₂ (the red trace) and O₂ (the blue trace) approximate arterial gas tensions. '
    'During hypercapnia, cerebral blood vessels dilate and cerebral blood-flow typically increases. '
    'During hypoxia, oxygen saturation of the blood falls to around 86%.':
        'Le gaz inspiré est manipulé pour augmenter le CO₂ (hypercapnie) et diminuer l’O₂ (hypoxie), tandis que '
        'la vélocité du flux sanguin cérébral est enregistrée par échographie Doppler transcrânienne et que la SpO₂, la '
        'saturation en oxygène, est mesurée à l’aide d’un oxymètre de pouls placé sur le doigt. '
        'Le CO₂ télé-expiratoire (le tracé rouge) et l’O₂ (le tracé bleu) approchent les pressions partielles des gaz artériels. '
        'Pendant l’hypercapnie, les vaisseaux sanguins cérébraux se dilatent et le flux sanguin cérébral augmente généralement. '
        'Pendant l’hypoxie, la saturation en oxygène du sang chute à environ 86 %.',

    # ---- glossary page ------------------------------------------------------
    'Supporting information': 'Informations complémentaires',
    'Glossary &amp; references': 'Glossaire et références',
    '09 / Reference': '09 / Référence',
    'Definitions and citations are provided here so the report can stand on its own when it is shared or printed.':
        'Les définitions et références sont fournies ici pour que le rapport soit autonome lorsqu’il est partagé ou imprimé.',
    '<h3 class="section-heading">Glossary</h3>': '<h3 class="section-heading">Glossaire</h3>',
    '<h3 class="section-heading">References</h3>': '<h3 class="section-heading">Références</h3>',
    'Montreal Cognitive Assessment.': 'Évaluation cognitive de Montréal.',
    'Forced expiratory volume in one second.': 'Volume expiratoire maximal en une seconde.',
    'Forced vital capacity.': 'Capacité vitale forcée.',
    'Peak expiratory flow.': 'Débit expiratoire de pointe.',
    '<strong>RR interval</strong><span>Time between consecutive R-waves on an ECG.</span>':
        '<strong>Intervalle RR</strong><span>Temps entre deux ondes R consécutives sur un ECG.</span>',
    'Root mean square of successive differences.': 'Racine carrée de la moyenne des différences successives.',
    '<strong>LF/HF ratio</strong>': '<strong>Rapport LF/HF</strong>',
    '<strong>E:I ratio</strong>': '<strong>Rapport E:I</strong>',
    'Ratio of low- to high-frequency HRV power.': 'Rapport de la puissance VFC basse fréquence sur haute fréquence.',
    'Arterial blood pressure.': 'Pression artérielle.',
    'Supine-to-stand test.': 'Test coucher-debout.',
    'Expiratory-to-inspiratory ratio.': 'Rapport expiratoire/inspiratoire.',
    '>Reminder<': '>Rappel<',
    'This research report is intended to share recorded study measurements with the participant. It is not a clinical diagnosis or treatment recommendation.':
        'Ce rapport de recherche vise à communiquer au participant les mesures enregistrées durant l’étude. Il ne constitue pas un diagnostic clinique ni une recommandation de traitement.',

    # ---- shared figure captions / alts -------------------------------------
    'R-aligned average of quality-screened resting beats': 'Moyenne alignée sur R des battements de repos retenus après contrôle qualité',
}


def translate(html: str) -> str:
    """Rewrite participant-facing English text to French (longest keys first)."""
    for en, fr in sorted(FR.items(), key=lambda kv: -len(kv[0])):
        html = html.replace(en, fr)
    return html
