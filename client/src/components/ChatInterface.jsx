import { useState } from "react"; // No need for useCallback
import axios from "axios"; // Ensure axios is imported
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import { useData } from "../context/DataContext"; // Import useData

const ChatInterface = () => {
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false); // State to manage loading
	const { data, setData } = useData(); // Get data and setData from context
	const messages = data.messages || []; // Access messages from context

	const handleSend = async () => {
		if (!input.trim()) return; // Prevent sending empty messages

		const inputData = { text: input }; // Use the actual input from the state
		setLoading(true); // Set loading to true when starting the API call

		try {
			const [
				categoryResponse,
				intentResponse,
				distilbertCategoryResponse,
				distilbertIntentResponse,
				gradientBoostingCategoryResponse,
				gradientBoostingIntentResponse,
			] = await Promise.all([
				axios.post("http://localhost:5000/baseline_category", inputData),
				axios.post("http://localhost:5000/baseline_intent", inputData),
				axios.post("http://localhost:5000/distilbert_category", inputData),
				axios.post("http://localhost:5000/distilbert_intent", inputData),
				axios.post(
					"http://localhost:5000/gradient_boosting_category",
					inputData
				),
				axios.post("http://localhost:5000/gradient_boosting_intent", inputData),
			]);

			// Create a bot response object
			const botResponse = {
				text: `The predicted category is "${categoryResponse.data.class}".`,
				categoryResponse: categoryResponse,
				intentResponse: intentResponse,
				distilbertCategoryResponse: distilbertCategoryResponse,
				distilbertIntentResponse: distilbertIntentResponse,
				gradientBoostingCategoryResponse: gradientBoostingCategoryResponse,
				gradientBoostingIntentResponse: gradientBoostingIntentResponse,
				sender: "bot",
				id: Date.now() + 1, // Unique ID for the bot response
			};
			// Create a message object for the user input
			const userMessage = {
				text: input,
				sender: "user",
				id: Date.now(), // Unique ID for the user message
				botResponse: botResponse,
			};

			// Update messages state in context
			const updatedMessages = [...messages, userMessage];

			// Update context with the new messages
			setData({ messages: updatedMessages }); // Store all relevant data in one local storage item
			setInput(""); // Clear input after sending
		} catch (error) {
			console.error("Error:", error);
		} finally {
			setLoading(false); // Set loading to false after the API call is complete
		}
	};

	return (
		<>
			{/* Loading Overlay */}
			{loading && (
				<div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50">
					<CircularProgress size={150} />
				</div>
			)}
			{/* Conditional rendering based on the length of messages in data */}
			{messages.length === 0 ? (
				<div className="border border-gray-300 rounded-lg shadow-lg w-[700px] px-6 py-10 bg-white">
					<div className="space-y-20 text-center">
						<h1 className="text-3xl font-extrabold tracking-wide text-gray-800">
							What Can I Help With?
						</h1>
						<InputSection
							input={input}
							setInput={setInput}
							handleSend={handleSend}
						/>
					</div>
				</div>
			) : (
				<div className="w-[700px] p-6 bg-white h-full">
					<div className="space-y-6 text-center justify-between flex flex-col h-full">
						<MessageDisplay /> {/* No need to pass messages as props */}
						<InputSection
							input={input}
							setInput={setInput}
							handleSend={handleSend}
						/>
					</div>
				</div>
			)}
		</>
	);
};

export default ChatInterface;
