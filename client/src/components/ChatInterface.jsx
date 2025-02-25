// components/ChatInterface.jsx
import React, { useState } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import ModelDropdown from "./dropdown";
import { useData } from "../context/DataContext";
import axios from "axios"; // Import Axios

const ChatInterface = () => {
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const { data, addMessage } = useData();

	// Get the active chat
	const activeChat = data.chatHistory.find(
		(chat) => chat.id === data.activeChatId
	);

	// Function to handle sending a message
	const handleSend = async () => {
		if (!input.trim()) return;

		setLoading(true);

		try {
			// Define API endpoints for both Baseline and Hybrid models
			const baselineApiUrl = "http://127.0.0.1:5000/baseline";
			const hybridApiUrl = "http://127.0.0.1:5000/hybrid";

			// Use Promise.all to fetch predictions from both models
			const [baselineResponse, hybridResponse] = await Promise.all([
				axios.post(baselineApiUrl, {
					text: input,
					max_length: 512,
					num_beams: 5,
					early_stopping: true,
				}),
				axios.post(hybridApiUrl, {
					text: input,
					max_length: 512,
					num_beams: 5,
					early_stopping: true,
				}),
			]);

			console.log("Baseline Response:", baselineResponse.data);
			console.log("Hybrid Response:", hybridResponse.data);

			// Combine responses into the desired structure
			const combinedResponse = {
				response: {
					baseline: baselineResponse.data,
					hybrid: hybridResponse.data,
				},
			};

			// Determine which model is selected (Baseline or Hybrid)
			const selectedModel = data.selectedModel || "Baseline"; // Default to Baseline

			// Select the generated text based on the selected model
			const generatedText =
				selectedModel === "Hybrid"
					? hybridResponse.data.generated_text
					: baselineResponse.data.generated_text;

			// Create the user message object
			const userMessage = {
				text: input,
				sender: "user",
				id: Date.now(),
				botResponse: {
					text: generatedText, // Only display the generated text from the selected model
					sender: "bot",
					id: Date.now() + 1,
					predictions: combinedResponse.response, // Save both Baseline and Hybrid predictions
					modelUsed: selectedModel, // Track which model was used
					weightedSum:
						selectedModel === "Hybrid"
							? hybridResponse.data.weighted_sum
							: baselineResponse.data.weighted_sum, // Add weighted sum
				},
			};

			// Add the message to the active chat
			addMessage(userMessage);

			// Clear the input field
			setInput("");
		} catch (error) {
			console.error("Error:", error);
			alert(`An error occurred: ${error.message}`);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="w-full h-full max-h-screen">
			{loading && (
				<div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50">
					<CircularProgress size={150} />
				</div>
			)}
			<div className="h-full px-36">
				<div className="flex flex-col h-full gap-5">
					{/* Model Dropdown */}
					<div className="flex justify-center">
						<ModelDropdown />
					</div>
					{/* Message Display */}
					<div className="flex-1 overflow-y-auto scrollbar-hide">
						<MessageDisplay />
					</div>
					{/* Input Section */}
					<InputSection
						input={input}
						setInput={setInput}
						handleSend={handleSend}
					/>
				</div>
			</div>
		</div>
	);
};

export default ChatInterface;
