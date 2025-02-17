import PropTypes from "prop-types";
import { Link } from "react-router-dom";
const BotMessage = ({ text, category, intent, ner, probabilities, id }) => {
	return (
		<div className="text-left">
			<div className="bg-[#CAF0F8] p-4 rounded-lg inline-block space-y-4 text-black border-l-[8px] border-[#00137F]">
				<div>
					<p>Category: {category}</p>
					<p>Intent: {intent}</p>
					<p>Named Entity Recognition: {ner}</p>
					<p>Response:</p>
					<p>{text}</p>
				</div>
				<div className="pb-2">
					<Link
						to={{
							pathname: `/result/${id}`,
						}}
						className=" text-gray-900 rounded-lg text-center hover:text-gray-600 transition duration-150 font-bold " // Add text-center for better alignment
					>
						View More &gt;
					</Link>
				</div>
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
