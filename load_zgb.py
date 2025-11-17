import os
from openai import OpenAI
from pinecone import Pinecone

# --- Konfiguration über Environment Variablen (Railway) ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ch-family-law")

print("DEBUG OPENAI_API_KEY:", repr(os.getenv("OPENAI_API_KEY")))
print("DEBUG ALL KEYS:", list(os.environ.keys()))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY nicht gesetzt")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY nicht gesetzt")

# OpenAI & Pinecone Clients initialisieren
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# --- ZGB Artikel 125 & 176 ---

ARTIKEL = [
    {
        "id": "zgb_125",
        "gesetz": "ZGB",
        "artikel": "125",
        "abschnitt": "E. Nachehelicher Unterhalt",
        "text_typ": "gesetzestext",
        "rechtsgebiet": "familienrecht",
        "text": """
Art. 125 ZGB – Nachehelicher Unterhalt

1 Ist einem Ehegatten nicht zuzumuten, dass er für den ihm gebührenden Unterhalt unter Einschluss einer angemessenen Altersvorsorge selbst aufkommt, so hat ihm der andere einen angemessenen Beitrag zu leisten.

2 Beim Entscheid, ob ein Beitrag zu leisten sei und gegebenenfalls in welcher Höhe und wie lange, sind insbesondere zu berücksichtigen:

1. die Aufgabenteilung während der Ehe;
2. die Dauer der Ehe;
3. die Lebensstellung während der Ehe;
4. das Alter und die Gesundheit der Ehegatten;
5. Einkommen und Vermögen der Ehegatten;
6. der Umfang und die Dauer der von den Ehegatten noch zu leistenden Betreuung der Kinder;
7. die berufliche Ausbildung und die Erwerbsaussichten der Ehegatten sowie der mutmassliche Aufwand für die berufliche Eingliederung der anspruchsberechtigten Person;
8. die Anwartschaften aus der eidgenössischen Alters- und Hinterlassenenversicherung und aus der beruflichen oder einer anderen privaten oder staatlichen Vorsorge einschliesslich des voraussichtlichen Ergebnisses der Teilung der Austrittsleistungen.

3 Ein Beitrag kann ausnahmsweise versagt oder gekürzt werden, wenn er offensichtlich unbillig wäre, insbesondere weil die berechtigte Person:

1. ihre Pflicht, zum Unterhalt der Familie beizutragen, grob verletzt hat;
2. ihre Bedürftigkeit mutwillig herbeigeführt hat;
3. gegen die verpflichtete Person oder eine dieser nahe verbundenen Person eine schwere Straftat begangen hat.
"""
    },
    {
        "id": "zgb_176",
        "gesetz": "ZGB",
        "artikel": "176",
        "abschnitt": "b. Regelung des Getrenntlebens",
        "text_typ": "gesetzestext",
        "rechtsgebiet": "familienrecht",
        "text": """
Art. 176 ZGB – Regelung des Getrenntlebens

1 Ist die Aufhebung des gemeinsamen Haushaltes begründet, so muss das Gericht auf Begehren eines Ehegatten:

1. die Unterhaltsbeiträge an die Kinder und den Unterhaltsbeitrag an den Ehegatten festlegen;
2. die Benützung der Wohnung und des Hausrates regeln;
3. die Gütertrennung anordnen, wenn es die Umstände rechtfertigen.

2 Diese Begehren kann ein Ehegatte auch stellen, wenn das Zusammenleben unmöglich ist, namentlich weil der andere es grundlos ablehnt.

3 Haben die Ehegatten minderjährige Kinder, so trifft das Gericht nach den Bestimmungen über die Wirkungen des Kindesverhältnisses die nötigen Massnahmen.
"""
    }
]


def embed(text: str):
    """Embedding mit OpenAI generieren."""
    emb = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return emb.data[0].embedding


def upload_articles():
    vectors = []

    for art in ARTIKEL:
        print(f"Erstelle Embedding für Art. {art['artikel']} ZGB ...")

        vec = embed(art["text"])

        metadata = {
            "text": art["text"],
            "quelle": "gesetz",
            "werk": "zgb",
            "gesetz": "ZGB",
            "artikel": art["artikel"],
            "abschnitt": art["abschnitt"],
            "text_typ": art["text_typ"],
            "rechtsgebiet": art["rechtsgebiet"],
            "chunk_index": 0,
        }

        vectors.append({
            "id": art["id"],
            "values": vec,
            "metadata": metadata
        })

    print(f"Lade {len(vectors)} Einträge in Pinecone hoch...")
    index.upsert(vectors=vectors)
    print("Fertig!")


if __name__ == "__main__":
    upload_articles()
