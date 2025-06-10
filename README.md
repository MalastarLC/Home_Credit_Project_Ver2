# Implémentez un modèle de scoring

## 1. Objectifs du Projet 

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

## 2. Découpage des Dossiers 

Ce projet est organisé de la manière suivante (les jeu de données ne sont pas dans le repository):

├── .github/workflows/ #Contient le ci.yml
│ └── ci.yml
│
├── mlruns/ #Dossiers contenant tout les modèles de prédiction
│ ├── Experiment 1
│ ├── Experiment ...
│ ├── Experiment x
│ └── Model registry
│
├── tests/ #contient les scripts choisis à exécuter dans ci.yml
│ ├── scripts for pytests
│
├── requirements.txt #Fichier listant les dépendances du projet
│
├── README.md # Ce fichier d'introduction
│
├── Projet_7_Analyse_Exploratoire_part : 1, 2 & 3.ipynb #Code ayant servi au traitement des données, création du modèle et analyse du data drift
│
├── API_Script.py #Script à exécuter pour lancer l'API localement
│
├── call_api_script.py #Script à exécuter pour tester l'API localement (penser à changer l'API endpoint URL pour tyest local)
│
├── pipeline_input_columns.txt #Colonnes attendues en entrée par le pipeline d'entrainement du modèle
│
└── preprocessing_pipeline.py #Fichier contenant les fonctions composant le pipeline de prétraitement des données et de prédiction du score