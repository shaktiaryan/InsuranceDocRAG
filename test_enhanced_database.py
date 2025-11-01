#!/usr/bin/env python3
"""
Test script for the enhanced Insurance Database with 4 tables
Tests all the new tools: customer_lookup, policy_details, claims_status, payment_history
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acs_agents.react_agent import insurance_react_agent

def test_enhanced_database():
    print("🚀 Testing Enhanced Insurance Database Tools\n")
    
    # Test scenarios
    test_cases = [
        {
            "name": "Customer Lookup by ID",
            "query": "Find customer with ID 1",
            "session_id": "test_001"
        },
        {
            "name": "Customer Lookup by Name", 
            "query": "Find customer Rajesh Sharma",
            "session_id": "test_002"
        },
        {
            "name": "Policy Details Lookup",
            "query": "Show details for policy IFL-TERM-001",
            "session_id": "test_003"
        },
        {
            "name": "Claims Status Check",
            "query": "Check status of claim CLM-2024-001",
            "session_id": "test_004"
        },
        {
            "name": "Payment History",
            "query": "Show payment history for customer 1",
            "session_id": "test_005"
        },
        {
            "name": "All Claims for Customer",
            "query": "Show all claims for customer 2",
            "session_id": "test_006"
        },
        {
            "name": "Policy Information",
            "query": "What is ULIP insurance?",
            "session_id": "test_007"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"{'='*60}")
        
        try:
            result = insurance_react_agent({
                "session_id": test_case["session_id"],
                "query": test_case["query"]
            })
            
            print(f"✅ Response:")
            print(f"{result['response']}")
            print(f"\n📊 Metadata:")
            metadata = result.get('metadata', {})
            print(f"- Session Duration: {metadata.get('session_duration', 0):.2f}s")
            print(f"- Tool Usage: {metadata.get('tool_usage', {})}")
            print(f"- Error Count: {metadata.get('error_count', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n")

if __name__ == "__main__":
    test_enhanced_database()