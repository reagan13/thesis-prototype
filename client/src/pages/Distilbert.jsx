import { useEffect, useState } from "react";
import Page from "./Page";
import { useData } from "../context/DataContext";
import { CatchingPokemonSharp } from "@mui/icons-material";
import { useParams } from "react-router-dom";

const names = ["Baseline", "Proposed Solution"];

const Distilbert = () => {
	const { id } = useParams();
	const { data } = useData();
	const messages = data.messages || [];
	const [botResponse, setBotResponse] = useState({});

	useEffect(() => {
		console.log("Enter");
		if (id) {
			const message = messages.find((msg) => msg.botResponse.id == id);
			if (message) {
				setBotResponse(message.botResponse);
			}
		}
	}, [data, id, botResponse, messages]);

	return (
		<div className="flex-grow grid grid-cols-1 md:grid-cols-2 gap-10 p-6">
			{names.map((name, index) => {
				// Conditional rendering based on the name
				if (name === "Baseline") {
					return (
						<Page
							key={index}
							categoryResponse={botResponse.categoryResponse}
							text={botResponse.text}
							intentResponse={botResponse.intentResponse}
						/>
					);
				} else if (name === "Proposed Solution") {
					return (
						<Page
							key={index}
							categoryResponse={botResponse.gradientBoostingCategoryResponse}
							text={botResponse.text}
							intentResponse={botResponse.gradientBoostingIntentResponse}
						/>
					);
				} else {
					// Optionally handle other cases or return null
					return null; // or return a default component
				}
			})}
		</div>
	);
};

export default Distilbert;
