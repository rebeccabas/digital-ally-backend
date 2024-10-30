from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import google.generativeai as genai
from datetime import datetime
from typing import Optional, Literal
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="Enhanced Support System and Complaint Generator")
genai.configure(api_key=os.getenv('API_KEY'))
model = genai.GenerativeModel('gemini-pro')

class Config:
    EMERGENCY_HOTLINE = os.getenv('EMERGENCY_HOTLINE', '1-800-799-SAFE')
    LEGAL_AID_URL = os.getenv('LEGAL_AID_URL', 'www.womenslaw.org')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')

class ComplaintInfo(BaseModel):
    filing_type: Literal["self", "third_party"]
    complainant_name: str
    complainant_address: str = Field(..., description="Full address including city and postal code")
    complainant_contact: Optional[str]
    complainant_email: Optional[str]
    victim_name: Optional[str]
    victim_address: Optional[str]
    relationship_to_victim: Optional[str]
    filing_authority: str = Field(..., description="Name and title of the filing authority")
    filing_authority_address: str = Field(..., description="Complete address of the filing authority")
    incident_details: str
    date_of_incident: str
    time_of_incident: str = Field(..., description="Approximate time of the incident")
    location_of_incident: str = Field(..., description="Specific location where the incident occurred")
    injuries_sustained: Optional[str]
    witness_information: Optional[str]
    evidence_description: Optional[str]

class SupportMessage(BaseModel):
    message: str

SUPPORT_CHAT_PROMPT = """
You are an empathetic support chatbot for people affected by domestic violence. Always:
1. Respond with care and understanding
2. Validate their feelings and experiences
3. Prioritize their safety and well-being
4. Provide emotional support first
5. Share information only if specifically asked
6. Use gentle, supportive language
7. Avoid judgmental statements
8. Remember this is a sensitive situation

User message: {message}

Provide a caring, supportive response:
"""

COMPLAINT_LETTER_TEMPLATE = """
{current_date}

{complainant_name}
{complainant_address}
{contact_section}

{filing_authority}
{filing_authority_address}

Subject: Formal Domestic Violence Complaint

Dear {authority_title},

I am writing to formally report a case of domestic violence that occurred on {incident_date} at {incident_time} in {incident_location}. {filing_capacity}

Incident Details:

{incident_narrative}

{injuries_section}

{evidence_section}

{witness_section}

Request for Action:

1. Implementation of immediate protective measures to ensure safety
2. Thorough investigation of the reported incidents
3. Appropriate legal action based on the investigation findings
4. Regular updates on the case progress
{additional_requests}

{confidentiality_statement}

Thank you for your immediate attention to this serious matter.

Respectfully,
{complainant_name}
{contact_details}

{victim_information}
"""

def create_contact_section(info: ComplaintInfo) -> str:
    contact_parts = []
    if info.complainant_email:
        contact_parts.append(info.complainant_email)
    if info.complainant_contact:
        contact_parts.append(info.complainant_contact)
    return "\n".join(contact_parts) if contact_parts else "Contact information provided separately"

def create_injuries_section(injuries: Optional[str]) -> str:
    if not injuries:
        return ""
    return f"""
Injuries Sustained:

{injuries}
"""

def format_evidence_description(evidence: Optional[str]) -> str:
    if not evidence:
        return "Documentation is available and can be provided upon request."
    
    evidence_points = evidence.split(", ")
    if len(evidence_points) == 1:
        return evidence
    
    return "\n".join(f"- {point}" for point in evidence_points)

@app.post("/api/generate-complaint")
async def generate_complaint_letter(complaint_info: ComplaintInfo):
    try:
        if complaint_info.filing_type == "third_party":
            if not all([complaint_info.victim_name, 
                       complaint_info.victim_address, 
                       complaint_info.relationship_to_victim]):
                raise HTTPException(
                    status_code=400,
                    detail="Third-party complaints require victim name, address, and relationship details"
                )
            
            filing_capacity = f"""I am filing this complaint on behalf of {complaint_info.victim_name}, 
as their {complaint_info.relationship_to_victim}. I have direct knowledge of the incidents 
described herein and am deeply concerned for their safety and well-being."""
            
            victim_information = f"""
Victim Information:
Name: {complaint_info.victim_name}
Address: {complaint_info.victim_address}
"""
        else:
            filing_capacity = "I am the direct victim of the incidents described in this complaint."
            victim_information = ""

        evidence_section = f"""
Supporting Evidence:
{format_evidence_description(complaint_info.evidence_description)}
"""

        witness_section = """
Witness Information:
""" + (complaint_info.witness_information or "Witness details can be provided as needed with appropriate privacy protections.")

        confidentiality_statement = """
I request that my personal information and the details of this complaint be handled with strict confidentiality 
to ensure my safety and protection."""

        authority_title = complaint_info.filing_authority.split()[-1] if len(complaint_info.filing_authority.split()) > 1 else "Sir/Madam"

        letter_content = COMPLAINT_LETTER_TEMPLATE.format(
            current_date=datetime.now().strftime("%B %d, %Y"),
            complainant_name=complaint_info.complainant_name,
            complainant_address=complaint_info.complainant_address,
            contact_section=create_contact_section(complaint_info),
            filing_authority=complaint_info.filing_authority,
            filing_authority_address=complaint_info.filing_authority_address,
            authority_title=authority_title,
            incident_date=complaint_info.date_of_incident,
            incident_time=complaint_info.time_of_incident,
            incident_location=complaint_info.location_of_incident,
            filing_capacity=filing_capacity,
            incident_narrative=complaint_info.incident_details,
            injuries_section=create_injuries_section(complaint_info.injuries_sustained),
            evidence_section=evidence_section,
            witness_section=witness_section,
            additional_requests="5. Provision of necessary support services and resources" if complaint_info.filing_type == "self" else "",
            confidentiality_statement=confidentiality_statement,
            contact_details=f"Contact: {complaint_info.complainant_contact}" if complaint_info.complainant_contact else "",
            victim_information=victim_information
        )

        return {
            "success": True,
            "complaint_letter": letter_content,
            "safety_reminder": "Please keep a copy of this document in a secure location.",
            "next_steps": [
                "Review all information for accuracy",
                "Make multiple copies for your records",
                "Consider seeking legal counsel",
                "Create a safety plan",
                "Keep all related documentation"
            ],
            "immediate_help": {
                "emergency": Config.EMERGENCY_HOTLINE,
                "legal_aid": Config.LEGAL_AID_URL
            }
        }

    except Exception as e:
        if Config.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail="We encountered an issue processing your request. If you're in immediate danger, please contact emergency services."
        )

@app.post("/api/support-chat")
async def get_support_chat(message: SupportMessage):
    try:
        prompt = SUPPORT_CHAT_PROMPT.format(message=message.message)
        response = model.generate_content(prompt)
        
        return {
            "response": response.text
        }

    except Exception as e:
        if Config.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail="We're having trouble processing your message. If you need immediate help, please call " + Config.EMERGENCY_HOTLINE
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=Config.DEBUG)