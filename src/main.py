from utils.functions import ask_neo


def main():
    print("🤖 Neo | Chatbot Diretor de Leads")
    print("Digite sua pergunta sobre as soluções de IA da NeuroSpark.")
    print("Para sair, digite: sair\n")

    while True:
        user_input = input("Você: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Neo: Até logo! 👋")
            break

        try:
            answer = ask_neo(user_input)
            print(f"\nNeo: {answer}\n")
        except Exception as e:
            print("\n[ERRO] Algo deu errado ao chamar o modelo.")
            print(f"Detalhes: {e}\n")


if __name__ == "__main__":
    main()
