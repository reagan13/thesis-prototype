import { useState } from "react";
import { Send } from "lucide-react";
import PropTypes from "prop-types";
import axios from "axios"; // Ensure axios is imported
import CircularProgress from "@mui/material/CircularProgress";

export const UserMessage = ({ text }) => (
	<div className="text-right">
		<div className="bg-blue text-white p-2 rounded-lg inline-block">{text}</div>
	</div>
);
UserMessage.propTypes = {
	text: PropTypes.string,
};

export const BotMessage = ({ text, category, intent, ner }) => {
	return (
		<div className="text-left">
			<div className="bg-gray-300 p-2 rounded-lg inline-block space-y-8">
				<div>
					<p>Category: {category}</p>
					<p>Intent: {intent}</p>
					<p>Named Entity Recognition: {ner}</p>
					<p>{text}</p>
				</div>
				<button className="bg-blue text-white p-2 rounded-lg">View More</button>
			</div>
		</div>
	);
};
BotMessage.propTypes = {
	text: PropTypes.string,
	category: PropTypes.string,
	intent: PropTypes.string,
	ner: PropTypes.string,
};

// MessageDisplay Component

export const MessageDisplay = ({ messages, category, intent, ner }) => (
	<div className="h-full overflow-y-auto max-h-[540px] border border-gray-300 rounded-lg p-10 space-y-5">
		{messages.map((message) => (
			<div key={message.id}>
				{message.sender === "user" ? (
					<UserMessage text={message.text} />
				) : (
					<BotMessage
						text={message.text}
						category={message.category}
						intent={message.intent}
						ner={message.ner}
					/>
				)}
			</div>
		))}
	</div>
);

MessageDisplay.propTypes = {
	messages: PropTypes.arrayOf(
		PropTypes.shape({
			id: PropTypes.number,
			text: PropTypes.string,
			sender: PropTypes.string,
		})
	),
};

// InputSection Component
const InputSection = ({ input, setInput, handleSend }) => (
	<div className="flex justify-between items-center gap-4">
		<input
			type="text"
			value={input}
			onChange={(e) => setInput(e.target.value)}
			placeholder="Enter your message..."
			className="rounded-full w-full border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500 py-4 px-5"
		/>
		<button
			onClick={handleSend}
			className="bg-blue-500 text-darkNavy rounded-full p-2 hover:bg-blue-600 transition duration-200"
		>
			<Send />
		</button>
	</div>
);
InputSection.propTypes = {
	input: PropTypes.string,
	setInput: PropTypes.func,
	handleSend: PropTypes.func,
};

const ChatInterface = () => {
	const [messages, setMessages] = useState([]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false); // State to manage loading

	const handleSend = async () => {
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

			// Log the responses
			console.log(
				"Gradient Boosting Category Response:",
				gradientBoostingCategoryResponse.data
			);
			console.log("Intent Response:", intentResponse.data);

			// Extract class and probabilities from the gradient boosting category response
			const { class: categoryClass } = gradientBoostingCategoryResponse.data;
			const { class: intentClass } = intentResponse.data; // Ensure this is correct

			// Simulated NER data (replace with actual extraction if available)
			const nerData = "This is a simulated NER";

			// Create a bot response
			const botResponse = {
				id: messages.length + 2,
				text: `The predicted category is "${categoryClass}".`,
				category: categoryClass,
				intent: intentClass,
				ner: nerData,
				sender: "bot",
			};

			console.log("Bot Response:", botResponse); // Log the bot response

			// Update messages state
			setMessages((prevMessages) => {
				const newMessages = [...prevMessages, botResponse];
				console.log("Updated Messages:", newMessages); // Log updated messages
				return newMessages;
			});
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
			{/* Conditional rendering based on messages */}
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
				<div className=" w-[700px] p-6 bg-white h-full">
					<div className="space-y-6 text-center justify-between flex flex-col h-full">
						<MessageDisplay messages={messages} />
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
