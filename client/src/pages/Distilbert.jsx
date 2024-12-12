import { useEffect, useState, useMemo } from "react";
import Page from "./Page";
import { useData } from "../context/DataContext";
import { useParams } from "react-router-dom";

const names = ["Baseline", "Proposed Solution"];

const Distilbert = () => {
	const { id } = useParams();
	const { data } = useData();

	const messages = useMemo(() => data.messages || [], [data]);
	const [botResponse, setBotResponse] = useState(null);

	// Debugging logs
	console.log("Current Data:", data);
	console.log("Current Messages:", messages);
	console.log("Current ID:", id);
	console.log("Current Bot Response:", botResponse);

	useEffect(() => {
		// Debugging function to log detailed information
		const findAndSetBotResponse = () => {
			console.log("Finding Bot Response - Start");
			console.log("Messages Length:", messages.length);
			console.log("Current ID:", id);

			// If no messages or no id, exit early
			if (!id || messages.length === 0) {
				console.log("No messages or ID found");
				return;
			}

			// Detailed logging of message finding process
			messages.forEach((msg, index) => {
				console.log(`Message ${index}:`, msg);
				console.log(`Message Bot Response ID:`, msg.botResponse?.id);
			});

			// Find the message
			const message = messages.find((msg) => {
				console.log("Comparing:", msg.botResponse?.id, "with", id);
				return msg.botResponse?.id == id;
			});

			if (message) {
				console.log("Found Message:", message);
				console.log("Bot Response:", message.botResponse);
				setBotResponse(message.botResponse);
			} else {
				console.log("No matching message found for ID:", id);
				setBotResponse(null);
			}
		};

		findAndSetBotResponse();
	}, [id, messages]);

	// Render method with comprehensive error handling
	const renderContent = () => {
		return names.map((name, index) => {
			// Baseline rendering
			if (name === "Baseline") {
				if (!botResponse) {
					return (
						<div key={index} className="text-red-500">
							No data available for Baseline
						</div>
					);
				}
				return (
					<Page
						key={index}
						name={name}
						categoryResponse={botResponse.categoryResponse}
						text={botResponse.text}
						intentResponse={botResponse.intentResponse}
					/>
				);
			}

			// Proposed Solution rendering
			if (name === "Proposed Solution") {
				if (!botResponse) {
					return (
						<div key={index} className="text-red-500">
							No data available for Proposed Solution
						</div>
					);
				}
				return (
					<Page
						key={index}
						name={name}
						categoryResponse={botResponse.gradientBoostingCategoryResponse}
						text={botResponse.text}
						intentResponse={botResponse.gradientBoostingIntentResponse}
					/>
				);
			}

			// Optionally handle other cases or return null
			return null; // or return a default component
		});
	};

	return (
		<>
			{!id ? (
				<div className="flex justify-center items-center h-full w-full">
					<h2>Please submit a message to view the results</h2>
				</div>
			) : (
				<div className="flex-grow grid grid-cols-1 md:grid-cols-2 gap-10 p-6">
					{renderContent()}
				</div>
			)}
		</>
	);
};

export default Distilbert;
