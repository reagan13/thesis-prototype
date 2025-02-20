import { useState } from "react";
import axios from "axios";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import { useData } from "../context/DataContext";

const ChatInterface = () => {
	const [input, setInput] = useState("set input sample");
	const [loading, setLoading] = useState(false);
	const { data, setData, isSidebarCollapsed } = useData(); // Get sidebar state
	const messages = data.messages || [];

	const handleSend = async () => {
		if (!input.trim()) return;
		const inputData = { text: input };
		setLoading(true);

		try {
			// const [categoryResponse] = await Promise.all([
			// 	axios.post("http://localhost:5000/baseline_category", inputData),
			// ]);

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
