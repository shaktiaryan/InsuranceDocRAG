#!/usr/bin/env python3
"""
Test script for the Insurance ReAct Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acs_agents.react_agent import insurance_react_agent

def test_agent():
    """Test the insurance ReAct agent with sample queries"""
    
    print("🚀 Testing Insurance ReAct Agent")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "session_id": "test_session_1",
            "query": "What is term life insurance?",
            "description": "General policy information query"
        },
        {
            "session_id": "test_session_2", 
            "query": "Show me policy details for CUST001",
            "description": "Customer-specific query"
        },
        {
            "session_id": "test_session_3",
            "query": "How do I file a claim?",
            "description": "Claims process query"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print("-" * 40)
        
        try:
            # Call the agent
            result = insurance_react_agent({
                "session_id": test_case["session_id"],
                "query": test_case["query"]
            })
            
            print(f"✅ Response: {result['response']}")
            print(f"📊 Metadata: {result['metadata']}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        print("-" * 40)
    
    print("\n🏁 Testing complete!")

if __name__ == "__main__":
    test_agent()