import PropTypes from "prop-types";
import { Link } from "react-router-dom";
const BotMessage = ({ text, category, intent, ner, probabilities, id }) => {
	return (
		<div className="text-left">
			<div className="bg-gray-300 p-2 rounded-lg inline-block space-y-8">
				<div>
					<p>Category: {category}</p>
					<p>Intent: {intent}</p>
					<p>Named Entity Recognition: {ner}</p>
					<p>{text}</p>
				</div>
				<Link
					to={{
						pathname: `/result/${id}`,
					}}
					className="bg-blue text-white p-2 rounded-lg text-center" // Add text-center for better alignment
				>
					View More
				</Link>
			</div>
		</div>
	);
};
BotMessage.propTypes = {
	text: PropTypes.string,
	category: PropTypes.string,
	intent: PropTypes.string,
	ner: PropTypes.string,
	id: PropTypes.number,
};

export default BotMessage;
