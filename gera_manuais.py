import openai
import json
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Voce escreve fragmentos curtos de manual medico tecnico.

Cada fragmento deve:
- Ter 3 a 5 frases
- Usar jargao medico tecnico (siglas como SCA, AVC, ITU, classes de medicamentos, criterios diagnosticos)
- Cobrir definicao, apresentacao clinica, diagnostico e conduta inicial
- Soar como livro-texto medico, nao como wikipedia ou divulgacao

Retorne SOMENTE o texto do fragmento, sem cabecalho, titulo ou numeracao."""

TOPICOS = [
    "migranea",
    "cefaleia tensional",
    "hipertensao arterial sistemica",
    "diabetes tipo 2",
    "asma bronquica",
    "pneumonia adquirida na comunidade",
    "sindrome coronariana aguda",
    "AVC isquemico",
    "infeccao urinaria baixa nao complicada",
    "doenca do refluxo gastroesofagico",
    "anemia ferropriva",
    "tromboembolismo pulmonar",
    "rinite alergica",
    "dermatite atopica",
    "transtorno de ansiedade generalizada",
    "depressao maior",
    "insonia cronica",
    "lombalgia mecanica",
    "conjuntivite viral",
    "otite media aguda",
    "hipotireoidismo primario",
    "vertigem posicional paroxistica benigna",
]


def gera(topico):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fragmento sobre: {topico}"},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    fragmentos = []
    for i, t in enumerate(TOPICOS):
        try:
            txt = gera(t)
            fragmentos.append({"id": i + 1, "texto": txt})
            print(f"{i+1}/{len(TOPICOS)}: {t}")
        except Exception as e:
            print(f"erro em {t}:", e)

    with open("manuais.jsonl", "w", encoding="utf-8") as f:
        for d in fragmentos:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{len(fragmentos)} fragmentos salvos em manuais.jsonl")
