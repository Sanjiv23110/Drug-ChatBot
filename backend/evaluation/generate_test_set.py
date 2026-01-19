"""
Simplified RAGas Test Set Generation

Creates a manual test set from predefined medical questions.
RAGas 0.4.2 has simplified APIs, so we'll create test questions manually
and use them for evaluation.
"""

import pandas as pd
from pathlib import Path

# Medical test questions covering key areas
# Format: (question, ground_truth_answer)
test_questions = [
    # Contraindications
    ("What are the contraindications for CeeNU?", 
     "CeeNU should not be given to individuals who have demonstrated a previous hypersensitivity to it or those with severe leukopenia and/or thrombocytopenia"),
    
    ("Who should not take TEVA-CHLORPROMAZINE?",
     "Patients with known hypersensitivity to chlorpromazine or related compounds"),
    
    # Drug Interactions  
    ("What is the interaction between valproic acid and CeeNU?",
     "Drug interaction information should be found in the drug interactions section of the monographs"),
    
    ("Can CeeNU be taken with other chemotherapy drugs?",
     "Information about combination therapy and drug interactions would be in the monographs"),
    
    # Dosage
    ("What is the recommended dosage of CeeNU?",
     "Dosage information should specify amount, frequency, and special considerations"),
    
    ("How should TEVA-CHLORPROMAZINE be administered?",
     "Administration route, dosage, and frequency information from monograph"),
    
    # Warnings and Precautions
    ("What are the main warnings for CeeNU use?",
     "Warnings about bone marrow suppression and other serious adverse effects"),
    
    ("Is CeeNU safe during pregnancy?",
     "Pregnancy safety information and contraindications for pregnant women"),
    
    # Indications
    ("What is CeeNU indicated for?",
     "Treatment of brain tumors and other approved indications"),
    
    ("What conditions is TEVA-CHLORPROMAZINE used to treat?",
     "Psychiatric conditions and other approved therapeutic uses"),
    
    # Special Populations
    ("Can elderly patients take CeeNU?",
     "Special dosing or precautions for elderly patients"),
    
    ("Is dose adjustment needed for patients with kidney disease?",
     "Renal impairment dosing adjustments and precautions"),
    
    # Adverse Reactions
    ("What are the common side effects of CeeNU?",
     "Hematologic toxicity, nausea, vomiting and other adverse reactions"),
    
    ("What serious adverse reactions should be monitored?",
     "Bone marrow suppression, pulmonary toxicity and other serious effects"),
    
    # Pharmacology
    ("How does CeeNU work?",
     "Mechanism of action as an alkylating agent"),
    
    ("What is the half-life of the drug?",
     "Pharmacokinetic parameters including elimination half-life"),
    
    # Storage
    ("How should CeeNU be stored?",
     "Storage temperature and stability requirements"),
    
    ("What is the shelf life of the medication?",
     "Expiration and stability information"),
    
    # Cross-Reference Testing (edge cases)
    ("What precautions are listed in the warnings section?",
     "Testing cross-referenced information between sections"),
    
    ("Are there any black box warnings?",
     "Serious warnings and precautions box information"),
    
    # Multi-context queries
    ("Can CeeNU cause long-term effects?",
     "Information about delayed toxicity and long-term risks"),
    
    ("What monitoring is required during CeeNU treatment?",
     "Required laboratory tests and clinical monitoring"),
    
    # Complex reasoning
    ("Should CeeNU dosage be adjusted for patients with both kidney and liver disease?",
     "Requires combining information about multiple organ impairments"),
    
    ("What are the differences in dosing between pediatric and adult patients?",
     "Age-specific dosing requirements and considerations"),
    
    # Specific drug information
    ("What is the active ingredient in the medication?",
     "Chemical name and active pharmaceutical ingredient"),
    
    ("What inactive ingredients are in the formulation?",
     "Excipients and non-active components"),
    
    # Safety
    ("What should be done in case of overdose?",
     "Overdose management and emergency procedures"),
    
    ("Are there any drug-food interactions?",
     "Dietary restrictions or food interaction warnings"),
    
    # Administration
    ("Can the medication be crushed or split?",
     "Tablet/capsule integrity requirements"),
    
    ("What should patients do if they miss a dose?",
     "Missed dose instructions and guidelines"),
]

def main():
    print("="*60)
    print("Creating Medical Test Set (Manual)")
    print("="*60)
    
    # Create DataFrame
    df = pd.DataFrame(test_questions, columns=['question', 'ground_truth'])
    
    print(f"\n✓ Created {len(df)} test questions")
    
    # Display categories
    categories = {
        'Contraindications': 2,
        'Drug Interactions': 2,
        'Dosage': 2,
        'Warnings/Precautions': 2,
        'Indications': 2,
        'Special Populations': 2,
        'Adverse Reactions': 2,
        'Pharmacology': 2,
        'Storage': 2,
        'Cross-References': 2,
        'Multi-context': 2,
        'Complex Reasoning': 2,
        'Drug Info': 2,
        'Safety': 2,
        'Administration': 2
    }
    
    print("\nTest Set Breakdown:")
    print("-" * 60)
    for category, count in categories.items():
        print(f"  {category:.<40} {count} questions")
    print("-" * 60)
    
    # Display sample questions
    print("\nSample Questions:")
    print("-" * 60)
    for i in range(min(5, len(df))):
        print(f"\n{i+1}. {df.iloc[i]['question']}")
    print("-" * 60)
    
    # Save to CSV
    output_path = Path("evaluation/medical_test_set.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved test set to: {output_path}")
    
    print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*60)
    print("Test set creation complete!")
    print("="*60)
    print("\nNext step: Run evaluation/evaluate_rag.py")

if __name__ == "__main__":
    main()
