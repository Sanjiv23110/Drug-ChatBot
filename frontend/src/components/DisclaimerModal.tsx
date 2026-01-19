import React from 'react';
import { ExternalLink } from 'lucide-react';

interface DisclaimerModalProps {
    isOpen: boolean;
    onAccept: () => void;
}

const DisclaimerModal: React.FC<DisclaimerModalProps> = ({ isOpen, onAccept }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
            <div className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl max-h-[90vh] overflow-y-auto p-8 shadow-2xl border border-gray-200 dark:border-gray-700">
                <h2 className="text-3xl font-bold text-red-600 dark:text-red-500 mb-6 flex items-center gap-2">
                    ⚠️ IMPORTANT MEDICAL DISCLAIMER
                </h2>

                <div className="space-y-5 text-gray-700 dark:text-gray-300">
                    <p className="font-semibold text-lg text-gray-900 dark:text-white">
                        Please Read Carefully Before Using This Tool
                    </p>

                    <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 dark:border-yellow-500 p-4 rounded-r-lg">
                        <p className="font-bold mb-2 text-yellow-900 dark:text-yellow-200">This is a reference tool ONLY</p>
                        <p className="text-yellow-800 dark:text-yellow-300">
                            This chatbot provides drug information from Health Canada monographs for
                            informational purposes only. It is NOT intended to replace professional
                            medical advice, diagnosis, or treatment.
                        </p>
                    </div>

                    <div className="space-y-3 text-sm">
                        <h3 className="font-bold text-base text-gray-900 dark:text-white">Terms of Use:</h3>
                        <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                            <li>Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions or medications</li>
                            <li>Never disregard professional medical advice or delay seeking it because of information from this chatbot</li>
                            <li>This tool does NOT check for drug interactions or patient-specific contraindications</li>
                            <li>Information provided may not reflect the most current product labeling</li>
                            <li>Verify all critical information with official sources before making clinical decisions</li>
                            <li>This tool is for use by licensed healthcare professionals only</li>
                        </ul>
                    </div>

                    <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 dark:border-blue-400 p-4 rounded-r-lg">
                        <p className="font-bold mb-2 text-blue-900 dark:text-blue-200 flex items-center gap-2">
                            📚 Official Drug Product Database
                        </p>
                        <p className="text-blue-800 dark:text-blue-300 mb-3">
                            For the most current and official drug information, please refer to Health Canada's Drug Product Database (DPD):
                        </p>
                        <a
                            href="https://health-products.canada.ca/dpd-bdpp/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium underline"
                        >
                            Visit Health Canada DPD
                            <ExternalLink className="w-4 h-4" />
                        </a>
                    </div>

                    <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 dark:border-red-400 p-4 rounded-r-lg">
                        <p className="font-bold mb-2 text-red-900 dark:text-red-200">⚠️ Liability Limitation:</p>
                        <p className="text-sm text-red-800 dark:text-red-300">
                            By using this tool, you acknowledge that the creators, developers, and distributors
                            are not liable for any errors, omissions, or clinical decisions made based on
                            information provided by this chatbot. Use at your own risk.
                        </p>
                    </div>

                    <div className="space-y-2 text-sm">
                        <h3 className="font-bold text-base text-gray-900 dark:text-white">Emergency Situations:</h3>
                        <p className="text-gray-700 dark:text-gray-300">
                            If you are experiencing a medical emergency, call 911 or your local emergency
                            number immediately. Do NOT rely on this chatbot for urgent medical guidance.
                        </p>
                    </div>

                    <p className="text-xs text-gray-500 dark:text-gray-400 italic border-t border-gray-200 dark:border-gray-700 pt-4">
                        Version 1.0 | Last Updated: January 2026
                    </p>
                </div>

                <div className="mt-8 flex gap-4">
                    <button
                        onClick={onAccept}
                        className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-4 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all text-lg"
                    >
                        I Understand and Agree
                    </button>
                </div>

                <p className="mt-4 text-xs text-center text-gray-500 dark:text-gray-400">
                    By clicking "I Understand and Agree", you confirm that you are a licensed healthcare
                    professional and accept the terms and limitations described above.
                </p>
            </div>
        </div>
    );
};

export default DisclaimerModal;
