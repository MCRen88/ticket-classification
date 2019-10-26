from constants import DATA_PATH, DATA_FT
import lorem
from random import randint
import json
import os

PROJECTS = (
    "Chalmers IT-avdelning",
    "Servicedesk",
    "Avslutade leveranser"
    "Certifikat",
    "Datorintro",
    "Fakturering",
    "Förvaltning",
    "Infoskärmar",
    "Infrastruktur",
    "Inköp",
    "Institutionsstöd",
    "Klient-plattform",
    "Lagring",
    "Larm",
    "Ledningsgruppen",
    "Licensinköp",
    "Närservice",
    "Närservice Lindholmen",
    "Närservice Nord",
    "Närservice Ost",
    "Närservice Väst",
    "Nät",
    "Print",
    "Sharepoint",
    "StuDAT",
    "Telebeställning",
    "Teletekniker",
    "Uppdrag",
    "Utveckling",
    "databas",
)

nb_classes = len(PROJECTS)

if os.path.exists(DATA_FT):
    os.remove(DATA_FT)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

NB_FILES = 10000
for i in range(NB_FILES):

    filename = f"data-{i}.json"
    filepath = os.path.join(DATA_PATH, filename)
    f = open(filepath, "w")

    project_name = PROJECTS[randint(0, nb_classes-1)]
    data = {
        "project": {
            "INFO_HEAD": project_name,
        },
        "title": lorem.sentence(),
        "description": lorem.text()
    }

    print(f"\rGenerating sample data: {i}/{NB_FILES}", end='')
    f.write(json.dumps(data))
    f.close()

print("\nDONE")
