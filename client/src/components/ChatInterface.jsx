import { useState } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import { useData } from "../context/DataContext";

const ChatInterface = () => {
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const { data, setData, setSelectedBotResponse } = useData(); // Get setSelectedBotResponse from context
	const messages = data.messages || [];

	const handleSend = async () => {
		console.log(input);
		if (!input.trim()) return;
		setLoading(true);

		try {
			// Generate two distinct responses
			const botResponse1 = {
				text: `Option 1: The predicted category is `,
				sender: "bot",
				id: Date.now() + 1,
				isSelected: false, // Track if this option is selected
			};

			const botResponse2 = {
				text: `Option 2: An alternative perspective is `,
				sender: "bot",
				id: Date.now() + 2,
				isSelected: false, // Track if this option is selected
			};

			const userMessage = {
				text: input,
				sender: "user",
				id: Date.now(),
				botResponses: [botResponse1, botResponse2], // Store both responses
			};

			setData({ messages: [...messages, userMessage] });
			setInput("");
		} catch (error) {
			console.error("Error:", error);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="w-full h-full max-h-screen">
			{loading && (
				<div className="flex items-center justify-center0 bg-black bg-opacity-50 backdrop-blur-sm z-50">
					<CircularProgress size={150} />
				</div>
			)}
			<div className="h-full px-36 ">
				<div className="flex flex-col h-full gap-5 ">
					<div className="flex-1 overflow-y-auto scrollbar-hide">
						<MessageDisplay />
					</div>
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
