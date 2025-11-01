from acs_agents.react_agent import insurance_react_agent

def run_insurance_agent():
    while True:
        try:
            user = input("Customer: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            response = insurance_react_agent({"session_id": "default", "query": user})
            print(f"\nIRIS(IFL): {response['response']}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    run_insurance_agent()