import os
import json
import math
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from .env file
load_dotenv()

# --- NEW: API Key Management ---
# Load all keys from the .env file and split them into a list
API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(',')
if not all(API_KEYS) or API_KEYS == ['']:
    raise ValueError("GOOGLE_API_KEYS environment variable not set or is empty. Please check your .env file.")

# --- NEW: Resilient LLM Invoker with Key Cycling ---
async def invoke_llm_with_retry(prompt_template, input_data):
    """
    Tries to invoke a LangChain chain with a list of API keys.
    If a key is rate-limited (ResourceExhausted), it automatically retries with the next key.
    """
    for i, key in enumerate(API_KEYS):
        try:
            # Initialize a new LLM instance with the current key for this specific call
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=key
                version="v1"
            )
            
            # Build the full chain using the prompt and the new LLM instance
            chain = prompt_template | llm | StrOutputParser()
            
            print(f"--- Attempting API call with Key #{i + 1} ---")
            response = await chain.ainvoke(input_data)
            print(f"--- Key #{i + 1} succeeded. ---")
            return response # Success, so we return the response

        except ResourceExhausted:
            print(f"Warning: API Key #{i + 1} is rate-limited or exhausted. Trying next key...")
            if i == len(API_KEYS) - 1: # If this was the last key in the list
                print("Error: All API keys are exhausted.")
                raise # Re-raise the final exception if all keys have failed
        except Exception as e:
            # Handle other potential errors (e.g., an invalid key format)
            print(f"An unexpected error occurred with Key #{i + 1}: {e}")
            if i == len(API_KEYS) - 1:
                raise
    
    # This line should ideally not be reached, but is a fallback.
    raise Exception("All API keys failed to generate a response.")


# --- Agent 1: The Analyst ---
analyst_template = """
You are a meticulous and data-driven financial analyst. Your primary role is to perform a rigorous, quantitative analysis of a user's financial situation and provide a structured, objective report. You must act as a "realist," grounding the user's ambitions in their current financial reality.

Your task is threefold:
Calculate Key Financial Ratios: Based on the raw user data, calculate the essential metrics that define their financial health.
Perform a Goal Sanity Check: Critically assess the feasibility of the user's stated goals against their calculated savings potential.
Synthesize Findings: Present all your calculations and observations in a clean, structured format for the next AI agent, The Strategist, to use.

CRITICAL INSTRUCTION: You MUST NOT provide any advice or recommendations. Your output is a factual, analytical report only.

USER DATA:
{user_data}

MARKET CONTEXT (For contextual understanding of risk and return):
{market_stats}

BEGIN ANALYSIS

PART 1: QUANTITATIVE FINANCIAL HEALTH ASSESSMENT
Monthly Savings Potential: (Calculate: Monthly Income - Monthly Expenses - Monthly Loan EMIs)
Savings Rate: (Calculate: [Monthly Savings Potential / Monthly Income] * 100)%
Debt-to-Income (DTI) Ratio: (Calculate: [Monthly Loan EMIs / Monthly Income] * 100)%
Emergency Fund Coverage: (Calculate: [Cash & Equivalents / Monthly Expenses]) months

PART 2: GOAL FEASIBILITY ANALYSIS (REALIST CHECK)
Overall Goal Assessment: [Provide a one-sentence summary of the goals' achievability. Use terms like "Highly Realistic," "Achievable with Discipline," "Ambitious," or "Requires Significant Re-evaluation."]
Goal-Specific Notes:
[For each goal, provide a brief comment. For example: "Goal 'Buy a Car' appears realistic within the user's timeline." or "Goal 'World Tour' is extremely ambitious and may conflict with the 'Buy a House' goal." If a goal seems impossible, state it directly.]

PART 3: SYNTHESIZED REPORT FOR THE STRATEGIST
User Profile Summary: A brief summary of the user (e.g., "A {{age}} -year-old with a {{risk_profile}} risk tolerance, a healthy savings rate, but significant high-interest debt.").

Key Strengths to Leverage:
[1-2 bullet points highlighting positive factors, like a high savings rate or low DTI.]

Urgent Red Flags to Address:
[1-2 bullet points on critical issues, like high-interest debt that needs immediate attention or an insufficient emergency fund.]

Asset Composition Insights:
[A brief comment on their current asset allocation. For example, "The user is heavily weighted in cash, indicating a conservative current stance despite an aggressive risk profile." or "The user has a good-sized equity portfolio, providing a strong base for future growth."]
"""
analyst_prompt = PromptTemplate(
    template=analyst_template,
    input_variables=['market_stats', 'user_data']
)

# --- Agent 2: The Strategist ---
strategist_template = """
You are a master financial strategist and tactician. You have received a detailed, quantitative report from your Analyst. Your sole purpose is to use this report to devise two distinct, high-level financial strategies.

Your task is threefold:
Critique and Confirm: Briefly acknowledge the Analyst's report and confirm you are proceeding based on its key findings (especially the Red Flags and Goal Assessment).
Devise the "Sentinel Plan": Create a conservative strategy that directly addresses the Analyst's red flags and aligns with a safety-first mindset.
Devise the "Voyager Plan": Create a growth-oriented strategy that leverages the user's strengths and risk tolerance, while still acknowledging the realism check.

CRITICAL INSTRUCTION: You must justify every strategic decision by explicitly referencing the metrics and observations from the Analyst's report. Do not output specific numbers or JSON. Your output is the core strategic logic for the Writer agent to follow.

ANALYST'S REPORT:
{analyst_summary}

BEGIN STRATEGIES

CONFIRMATION:
[Start with a single sentence confirming you have reviewed the analyst's report. For example: "Analyst's report received and validated. Proceeding with strategy formulation based on the user's {{Financial Health Score}} and {{Overall Goal Assessment}}."]

1. THE SENTINEL PLAN (A Strategy of SECURITY and Stability)

Core Philosophy: [Explain the plan's philosophy in one sentence, directly referencing the user's situation. Example: "This strategy prioritizes eliminating the user's high-interest debt and building a robust emergency fund, creating a secure foundation before focusing on long-term, low-risk growth."]

Key Priorities (in order):
Debt Management: [State the strategic approach. Example: "Given the {{DTI Ratio}}, the primary focus is to aggressively pay down the high-interest debt identified by the Analyst."]
Emergency Fund: [State the goal. Example: "Based on the Analyst's finding of {{Emergency Fund Coverage}} months, the next priority is to build this up to a 6-month cushion."]
Investment Approach: [Define the investment style. Example: "Once the foundation is secure, begin steady, low-cost investing. The asset allocation should be heavily weighted towards low-volatility assets like bonds, using the market data's correlation matrix to ensure diversification away from equities."]
Goal Alignment: [Comment on how this plan affects goals. Example: "This approach will likely result in longer timelines for the user's stated goals, aligning with the Analyst's 'Ambitious' feasibility assessment, but it significantly increases the probability of success."]

2. THE VOYAGER PLAN (A Strategy of disciplined GROWTH)

Core Philosophy: [Explain the plan's philosophy. Example: "This strategy leverages the user's {{High Savings Rate/Aggressive Risk Profile}} to pursue faster growth, accepting higher market volatility to potentially reach their goals sooner."]

Key Priorities (in order):
Debt Management: [State the approach. Example: "While still addressing the high-interest debt, this plan allocates a larger portion of the monthly savings towards immediate investment to capitalize on market returns."]
Emergency Fund: [State the goal. Example: "A more agile 3-month emergency fund is sufficient, reflecting the user's high-risk tolerance."]
Investment Approach: [Define the investment style. Example: "The asset allocation will be heavily weighted towards equities, as suggested by their historical high returns in the market data. A small, tactical allocation to high-risk assets like crypto can be considered, directly aligning with the user's risk profile."]
Goal Alignment: [Comment on how this plan affects goals. Example: "This approach provides a more aggressive timeline for achieving the user's goals, but the user must be prepared for the higher volatility and potential drawdowns noted in the market analysis."]
"""
strategist_prompt = PromptTemplate(
    template=strategist_template,
    input_variables=["analyst_summary"]
)

# --- Agent 3: The Writer ---
writer_template = """
You are an expert financial writer and quantitative analyst. You have been given a user's complete financial profile and two high-level strategies from a master strategist. Your sole task is to translate these strategies into a detailed, user-friendly financial plan, performing specific calculations as instructed.

CRITICAL INSTRUCTIONS:
1. You MUST adhere strictly to the philosophy and action steps outlined in the provided strategies.
2. You MUST calculate the projected goal timelines using the provided financial projection functions. Show your work.
3. Your entire response MUST be a single, valid JSON object, with no other text, comments, or explanations before or after it.

USER DATA:
{user_data}

HIGH-LEVEL STRATEGIES FROM THE STRATEGIST:
{strategies}

FINANCIAL CALCULATION FUNCTIONS (for your internal use):
def project_goal_timeline(target_amount, initial_investment, monthly_contribution, annual_return_rate):
    # This function calculates the number of years to reach a financial goal.
    # Use the avg_annual_return_percent from the user's market_stats.json for the annual_return_rate.
    pass
    
BEGIN FINAL PLAN GENERATION

Based on all the provided information, create the complete financial plan.

For the projected_goal_timeline_years:
1. Use the project_goal_timeline function.
2. For the annual_return_rate, use the appropriate blended rate based on the asset allocation for each plan. For example, for the Sentinel plan, you might use a blended rate of (30% * equity_return) + (50% * bond_return) + (10% * commodities_return) + (10% * cash_return).

For the recommendations:
Translate the "Key Priorities" from the Strategist's report into 3-4 clear, actionable, and encouraging steps for the user.

OUTPUT FORMAT (Strictly adhere to this JSON Structure):
{{
  "sentinel_plan": {{
    "summary": "A brief, encouraging summary of this safe plan, directly reflecting the Strategist's philosophy.",
    "asset_allocation": {{ "equities": "X%", "bonds": "Y%", "commodities": "Z%", "cash": "A%" }},
    "projected_goal_timeline_years": {{ "User's Goal Name 1": "X", "User's Goal Name 2": "Y" }},
    "recommendations": ["Detailed recommendation 1", "Detailed recommendation 2", "Detailed recommendation 3"]
  }},
  "voyager_plan": {{
    "summary": "A brief, encouraging summary of this growth-oriented plan, directly reflecting the Strategist's philosophy.",
    "asset_allocation": {{ "equities": "X%", "bonds": "Y%", "crypto": "Z%", "cash": "A%" }},
    "projected_goal_timeline_years": {{ "User's Goal Name 1": "X", "User's Goal Name 2": "Y" }},
    "recommendations": ["Detailed recommendation 1", "Detailed recommendation 2", "Detailed recommendation 3"]
  }}
}}
"""
writer_prompt = PromptTemplate(
    template=writer_template,
    input_variables=["user_data", "strategies"]
)

# --- The AI Assembly Line Chain ---
async def generate_plan_with_assembly_line(user_profile: dict):
    with open("market_stats.json", "r") as f:
        market_stats = json.load(f)

    analyst_input = {"user_data": json.dumps(user_profile), "market_stats": json.dumps(market_stats)}
    analyst_summary = await invoke_llm_with_retry(analyst_prompt, analyst_input)

    strategist_input = {"analyst_summary": analyst_summary}
    strategies = await invoke_llm_with_retry(strategist_prompt, strategist_input)

    writer_input = {"user_data": json.dumps(user_profile), "strategies": strategies}
    final_plan_str = await invoke_llm_with_retry(writer_prompt, writer_input)
    
    return final_plan_str

# --- Agent 4: The Economic Forecaster (Personalized Storyteller) ---

forecaster_template = """
You are an expert economic forecaster and financial storyteller for the Indian market.
Your task is to generate 3 distinct, plausible economic scenarios for the next 5 years based on the user's primary financial goal.
One scenario must be optimistic, one pessimistic, and one mixed/neutral.

CRITICAL INSTRUCTION: When writing each narrative, you MUST subtly weave in a reference to how this economic climate might affect the user's primary goal.

USER'S PRIMARY GOAL: {user_goal}

For EACH of the 3 scenarios, you MUST provide two things:
1. A short, creative 'narrative' paragraph describing the economic story and its potential impact on the user's goal.
2. A structured JSON object with these exact keys and estimated annual percentage values: {{"avg_equity_return": X, "avg_bond_return": Y, "avg_inflation": Z}}

Your entire response MUST be a single, valid JSON object, with no other text before or after it.
The JSON object should follow this exact structure:
{{
  "scenarios": [
    {{
      "name": "The Optimistic Scenario",
      "narrative": "A descriptive story of a booming economy that could accelerate your goal of {user_goal}...",
      "parameters": {{ "avg_equity_return": 18.0, "avg_bond_return": 7.5, "avg_inflation": 4.5 }}
    }},
    {{
      "name": "The Pessimistic Scenario",
      "narrative": "A descriptive story of a sluggish economy that might pose challenges to your goal of {user_goal}...",
      "parameters": {{ "avg_equity_return": 2.0, "avg_bond_return": 6.0, "avg_inflation": 8.0 }}
    }},
    {{
      "name": "The Neutral Scenario",
      "narrative": "A descriptive story of a mixed economy with specific opportunities and risks for your goal of {user_goal}...",
      "parameters": {{ "avg_equity_return": 9.0, "avg_bond_return": 6.5, "avg_inflation": 6.0 }}
    }}
  ]
}}
"""
forecaster_prompt = PromptTemplate(
    template=forecaster_template,
    input_variables=["user_goal"]
)

def project_goal_timeline(target_amount, initial_investment, monthly_contribution, annual_return_rate):
    """
    Calculates the number of years to reach a financial goal.
    (This function is unchanged)
    """
    if monthly_contribution <= 0 and initial_investment < target_amount:
        return float('inf')

    if annual_return_rate <= 0:
        if monthly_contribution <= 0:
            return float('inf')
        # Avoid division by zero if target is already met
        if (target_amount - initial_investment) <= 0:
            return 0.0
        return (target_amount - initial_investment) / (monthly_contribution * 12)

    monthly_rate = (1 + annual_return_rate) ** (1/12) - 1
    
    if abs(monthly_rate) < 1e-9: # Handles cases where annual_return_rate is extremely small
        if monthly_contribution <= 0:
            return float('inf')
        if (target_amount - initial_investment) <= 0:
            return 0.0
        return (target_amount - initial_investment) / (monthly_contribution * 12)

    # Formula for number of periods (months) in an annuity
    # n = log( (FV*r + PMT) / (P*r + PMT) ) / log(1+r)
    try:
        # We add a small epsilon to avoid log(0) if PMT is 0 and FV equals P
        numerator = math.log((target_amount * monthly_rate + monthly_contribution + 1e-9))
        denominator = math.log((initial_investment * monthly_rate + monthly_contribution + 1e-9))
        log_1_plus_r = math.log(1 + monthly_rate)
        
        if log_1_plus_r == 0: # Should be caught by earlier check, but as a safeguard
             return float('inf')

        months = (numerator - denominator) / log_1_plus_r
    except (ValueError, ZeroDivisionError):
        # Fallback to iteration if formula fails (e.g., due to negative logs)
        current_value = float(initial_investment)
        months = 0
        while current_value < target_amount:
            interest = current_value * monthly_rate
            current_value += interest + monthly_contribution
            months += 1
            if months > 1200: # 100 years
                return float('inf')
        return round(months / 12, 1)

    return round(months / 12, 1)


async def run_economic_forecaster(user_profile: dict):
    """
    Runs the personalized economic forecaster agent.
    """
    # Extract the user's primary goal (we'll assume the first one listed)
    primary_goal = user_profile['goals'][0]['name'] if user_profile['goals'] else "achieving their financial targets"

    # Invoke the LLM with the user's goal for personalization
    scenarios_str = await invoke_llm_with_retry(forecaster_prompt, {"user_goal": primary_goal})
    
    # Robust JSON parsing
    start_index = scenarios_str.find('{')
    end_index = scenarios_str.rfind('}') + 1
    if start_index == -1 or end_index == 0:
        raise ValueError("Forecaster agent returned invalid data (no JSON object found).")
    json_str = scenarios_str[start_index:end_index]
    scenarios_data = json.loads(json_str)

    # Calculate the user's available monthly savings for investment
    monthly_savings = user_profile['monthly_income'] - user_profile['monthly_expenses'] - user_profile['liabilities']['loans_emi']
    initial_investment = user_profile['assets']['equity_investments'] + user_profile['assets']['other_investments']

    # Loop through each AI-generated scenario and calculate the new goal timelines
    for scenario in scenarios_data['scenarios']:
        params = scenario['parameters']
        # Create a blended return rate based on a moderate (60/40) portfolio for simulation
        blended_return = (params['avg_equity_return'] * 0.60 + params['avg_bond_return'] * 0.40) / 100

        projected_timelines = {}
        for goal in user_profile['goals']:
            timeline = project_goal_timeline(
                target_amount=float(goal['target_amount']),
                initial_investment=float(initial_investment),
                monthly_contribution=float(monthly_savings),
                annual_return_rate=float(blended_return)
            )
            projected_timelines[goal['name']] = "More than 100 years" if math.isinf(timeline) else f"{timeline} years"
        
        scenario['projected_timelines'] = projected_timelines
        
    return scenarios_data


# --- Agent 5: The Context-Aware Q&A Agent ---
qa_template = """
You are FinPilot, an expert, patient, and educational financial co-pilot.
A user is asking a follow-up question about the financial plan you have already created for them.
Your goal is to provide high-quality, reasonable, and informative answers that build the user's confidence and financial literacy.

Core Instructions:
1.  Strict Context Adherence: Your entire answer MUST be based ONLY on the User's Profile, the Generated Plan, and the Chat History provided below. Do not make up any external information.
2.  Educational Mandate: If the user asks for a definition of a financial term (e.g., 'diversification', 'asset allocation', 'index fund'), explain it in simple, layman's terms, perhaps using a short analogy.
3.  Justification Mandate: When the user asks "why" a recommendation was made, you MUST deduce the reasoning by linking their specific profile data (e.g., age, risk profile, goals, high-interest debt) to the recommendation using general financial principles. You must act as if you were the original strategist who made these decisions.
4.  Safety and Boundaries: NEVER give specific stock, bond, or product recommendations (e.g., "buy Reliance stock"). Only discuss asset classes and strategies (e.g., "invest in diversified equity mutual funds"). If asked for a specific product, politely decline and explain that you can only provide strategic guidance.
5.  No Deflection: NEVER tell the user to "look up the internet" or that you cannot answer. Always provide a helpful response based on the provided context and your embedded financial knowledge.

CONTEXT - USER'S ORIGINAL PROFILE:
{user_profile}

CONTEXT - THE GENERATED PLAN:
{generated_plan}

CONTEXT - RECENT CHAT HISTORY:
{chat_history}

Based on all of this context, answer the user's question.

USER'S QUESTION:
{new_question}

YOUR ANSWER:
"""
qa_prompt = PromptTemplate.from_template(qa_template)

async def run_qa_agent(payload: dict):
    qa_input = {
        "user_profile": json.dumps(payload['userProfile']),
        "generated_plan": json.dumps(payload['generatedPlan']),
        "chat_history": json.dumps(payload['chatHistory']),
        "new_question": payload['newQuestion']
    }
    answer = await invoke_llm_with_retry(qa_prompt, qa_input)
    return {"response": answer}
