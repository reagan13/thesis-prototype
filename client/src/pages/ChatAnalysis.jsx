import { useParams } from "react-router-dom";
import BarChart from "../components/BarChart"; // Import the reusable BarChart component
import Modal from "../components/Modal"; // Import the Modal component
import { useData } from "../context/DataContext"; // Import useData
import { useState } from "react"; // Import useState

const ChatAnalysis = () => {
	const { id } = useParams(); // Get the message ID from the URL
	const { data } = useData(); // Access global state
	const [isModalOpen, setIsModalOpen] = useState(false);
	const [selectedBox, setSelectedBox] = useState(null);

	// Find the active chat and the specific message by messageId
	const activeChat = data.chatHistory.find(
		(chat) => chat.id === data.activeChatId
	);
	if (!activeChat) {
		return <div className="p-6 text-center">No active chat found.</div>;
	}

	const message = activeChat.messages.find((msg) => msg.id === Number(id));
	if (!message || !message.botResponse) {
		return <div className="p-6 text-center">No message found.</div>;
	}

	// Handlers for each box click
	const handleBoxClick = (boxType) => {
		setSelectedBox(boxType); // Set the selected box type
		setIsModalOpen(true); // Open the modal
	};

	// Helper function to calculate time duration
	const calculateDuration = (startTime, endTime) => {
		const start = new Date(startTime);
		const end = new Date(endTime);
		const durationMs = end - start;
		const seconds = Math.floor(durationMs / 1000);
		return `${seconds} seconds`;
	};

	return (
		<div className="p-6  max-h-screen overflow-y-auto scrollbar-hide scroll-hide  h-full">
			{/* Main Title */}
			<h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
				Chat Analysis
			</h1>

			{/* User Query Section */}
			<div className="mb-8 p-6 bg-white rounded-lg shadow-md">
				<p className="text-lg text-gray-700">
					<strong>User Query:</strong>{" "}
					<span className="font-semibold text-gray-900">{message.text}</span>
				</p>
			</div>

			{/* Response Generation Section */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
				{/* Baseline Response */}
				<div className="p-6 bg-white rounded-lg shadow-md">
					<h2 className="text-xl font-semibold text-gray-800 mb-4">
						Baseline Response
					</h2>
					<p className="text-gray-700 mb-4">
						<strong>Generated Text:</strong>{" "}
						{message.botResponse.predictions?.baseline?.generated_text || "N/A"}
					</p>
					<div className="space-y-2 text-sm text-gray-600">
						<p>
							<strong>Category:</strong>{" "}
							{message.botResponse.predictions?.baseline?.baseline_predictions
								?.category?.label ||
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.category?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.category?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>Intent:</strong>{" "}
							{message.botResponse.predictions?.baseline?.baseline_predictions
								?.intent?.label ||
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.intent?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.intent?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>NER:</strong>{" "}
							{Array.isArray(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.ner
							) &&
							message.botResponse.predictions?.baseline?.baseline_predictions
								?.ner.length > 0
								? message.botResponse.predictions?.baseline?.baseline_predictions?.ner.map(
										(entity, index) => (
											<span key={index} className="mr-2">
												{"Text: "}
												{entity.text},{" Type: "}
												{entity.type} (Confidence:{" "}
												{(entity.confidence * 100).toFixed(2)}%)
											</span>
										)
								  )
								: "None"}
						</p>
						<p>
							<strong>Weighted Sum:</strong>{" "}
							{message.botResponse.predictions?.baseline?.weighted_sum.toFixed(
								4
							)}
						</p>
						<p>
							<strong>Processing Time:</strong>{" "}
							{calculateDuration(
								message.botResponse.predictions?.baseline?.start_time,
								message.botResponse.predictions?.baseline?.end_time
							)}
						</p>
					</div>
				</div>

				{/* Hybrid Response */}
				<div className="p-6 bg-white rounded-lg shadow-md">
					<h2 className="text-xl font-semibold text-gray-800 mb-4">
						Hybrid Response
					</h2>
					<p className="text-gray-700 mb-4">
						<strong>Generated Text:</strong>{" "}
						{message.botResponse.predictions?.hybrid?.generated_text || "N/A"}
					</p>
					<div className="space-y-2 text-sm text-gray-600">
						<p>
							<strong>Category:</strong>{" "}
							{message.botResponse.predictions?.hybrid?.hybrid_predictions
								?.category?.label ||
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.category?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.category?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>Intent:</strong>{" "}
							{message.botResponse.predictions?.hybrid?.hybrid_predictions
								?.intent?.label ||
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.intent?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.intent?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>NER:</strong>{" "}
							{Array.isArray(
								message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner
							) &&
							message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner
								.length > 0
								? message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner.map(
										(entity, index) => (
											<span key={index} className="mr-2">
												{"Text: "}
												{entity.text},{" Type: "}
												{entity.type} (Confidence:{" "}
												{(entity.confidence * 100).toFixed(2)}%)
											</span>
										)
								  )
								: "None"}
						</p>
						<p>
							<strong>Weighted Sum:</strong>{" "}
							{message.botResponse.predictions?.hybrid?.weighted_sum.toFixed(4)}
						</p>
						<p>
							<strong>Processing Time:</strong>{" "}
							{calculateDuration(
								message.botResponse.predictions?.hybrid?.start_time,
								message.botResponse.predictions?.hybrid?.end_time
							)}
						</p>
					</div>
				</div>
			</div>

			{/* Modal */}
			{/* {isModalOpen && (
                <Modal
                    selectedBox={selectedBox}
                    baselineData={message.botResponse.predictions?.baseline}
                    hybridData={message.botResponse.predictions?.hybrid}
                    onClose={() => setIsModalOpen(false)} // Close modal handler
                />
            )} */}
		</div>
	);
};

export default ChatAnalysis;
