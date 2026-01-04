import React from 'react';

interface DisclaimerModalProps {
    onAccept: () => void;
}

const DisclaimerModal: React.FC<DisclaimerModalProps> = ({ onAccept }) => {
    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-2xl max-h-[90vh] overflow-y-auto p-6 shadow-xl">
                <h2 className="text-2xl font-bold text-red-600 mb-4">⚠️ IMPORTANT MEDICAL DISCLAIMER</h2>

                <div className="space-y-4 text-gray-700">
                    <p className="font-semibold text-lg">
                        Please Read Carefully Before Using This Tool
                    </p>

                    <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                        <p className="font-bold mb-2">This is a reference tool ONLY</p>
                        <p>
                            This chatbot provides drug information from Health Canada monographs for
                            informational purposes only. It is NOT intended to replace professional
                            medical advice, diagnosis, or treatment.
                        </p>
                    </div>

                    <div className="space-y-2 text-sm">
                        <h3 className="font-bold text-base">Terms of Use:</h3>
                        <ul className="list-disc pl-6 space-y-1">
                            <li>Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions or medications</li>
                            <li>Never disregard professional medical advice or delay seeking it because of information from this chatbot</li>
                            <li>This tool does NOT check for drug interactions or patient-specific contraindications</li>
                            <li>Information provided may not reflect the most current product labeling</li>
                            <li>Verify all critical information with official sources before making clinical decisions</li>
                            <li>This tool is for use by licensed healthcare professionals only</li>
                        </ul>
                    </div>

                    <div className="bg-red-50 border-l-4 border-red-500 p-4">
                        <p className="font-bold mb-2">⚠️ Liability Limitation:</p>
                        <p className="text-sm">
                            By using this tool, you acknowledge that the creators, developers, and distributors
                            are not liable for any errors, omissions, or clinical decisions made based on
                            information provided by this chatbot. Use at your own risk.
                        </p>
                    </div>

                    <div className="space-y-2 text-sm">
                        <h3 className="font-bold text-base">Emergency Situations:</h3>
                        <p>
                            If you are experiencing a medical emergency, call 911 or your local emergency
                            number immediately. Do NOT rely on this chatbot for urgent medical guidance.
                        </p>
                    </div>

                    <p className="text-xs text-gray-500 italic">
                        Version 1.0 | Last Updated: December 2024
                    </p>
                </div>

                <div className="mt-6 flex gap-4">
                    <button
                        onClick={onAccept}
                        className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-semibold"
                    >
                        I Understand and Agree
                    </button>
                </div>

                <p className="mt-4 text-xs text-center text-gray-500">
                    By clicking "I Understand and Agree", you confirm that you are a licensed healthcare
                    professional and accept the terms and limitations described above.
                </p>
            </div>
        </div>
    );
};

export default DisclaimerModal;
