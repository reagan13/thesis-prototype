import { useState } from "react";
import SearchInput from "../components/SearchInput"; // Assuming you have a SearchInput component

const ChatbotInterface = () => {
	const [messages, setMessages] = useState([]);
	const [input, setInput] = useState("");
	const [categories, setCategories] = useState([]);
	const [analysisReport, setAnalysisReport] = useState(""); // For analysis report

	const handleSend = () => {
		if (input.trim()) {
			// Add user message
			setMessages([...messages, { text: input, sender: "user" }]);
			// Simulate a response (replace with actual logic)
			const response = "This is a simulated response.";
			setMessages((prev) => [...prev, { text: response, sender: "bot" }]);
			// Simulate categories (replace with actual logic)
			setCategories([
				{ name: "Category 1", probability: 0.8 },
				{ name: "Category 2", probability: 0.2 },
				{ name: "Category 3", probability: 0.5 },
			]);
			// Simulate analysis report (replace with actual logic)
			setAnalysisReport(
				"The analysis indicates that the input is primarily related to Category 1 with a high probability of 80%. " +
					"Category 2 has a lower relevance at 20%, while Category 3 shows a moderate relevance of 50%. " +
					"This suggests that the user's input is likely focused on the themes associated with Category 1."
			);
			setInput(""); // Clear input
		}
	};

	return (
		<div className="grid grid-cols-3 gap-4 p-4 h-full">
			<div className="col-span-2 flex flex-col h-full">
				<SearchInput /> {/* Existing search input component */}
				<div className="border border-gray-300 rounded-lg shadow-lg mt-4 flex flex-col flex-grow">
					<div className="bg-green-500 text-white p-4 text-center rounded-t-lg">
						Chatbot
					</div>
					<div className="flex-1 p-4 overflow-y-auto bg-gray-100">
						{/* Conversation area */}
						{messages.map((msg, index) => (
							<div
								key={index}
								className={`mb-2 p-2 rounded ${
									msg.sender === "user"
										? "bg-blue-200 self-end"
										: "bg-green-200"
								}`}
							>
								{msg.text}
							</div>
						))}
					</div>
					<div className="flex p-2">
						<input
							type="text"
							className="border-none p-2 rounded-l-lg focus:outline-none w-full"
							placeholder="Type your message..."
							value={input}
							onChange={(e) => setInput(e.target.value)}
						/>
						<button
							onClick={handleSend}
							className="bg-blue-500 text-white p-2 rounded-r-lg"
						>
							Submit
						</button>
					</div>
				</div>
			</div>
			<div className="border border-gray-300 rounded-lg shadow-lg p-4 h-full">
				{/* Categorization section */}
				<h3 className="font-bold">Categorization</h3>
				<ul>
					{categories.map((category, index) => (
						<li key={index} className="flex justify-between">
							<span>{category.name}</span>
							<span>{(category.probability * 100).toFixed(0)}%</span>
						</li>
					))}
				</ul>
			</div>
			<div className="border border-gray-300 rounded-lg shadow-lg p-4 h-full">
				{/* Analysis report section */}
				<h3 className="font-bold">Analysis Report</h3>
				<p>{analysisReport}</p>
			</div>
		</div>
	);
};

export default ChatbotInterface;
