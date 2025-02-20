import PropTypes from "prop-types";
import { Link } from "react-router-dom";

const BotMessage = ({
	text,
	category,
	intent,
	ner,
	id,
	isSelected,
	onSelect,
}) => {
	return (
		<div className="p-3 rounded-3xl inline-block space-y-4 text-left max-w-[500px] border border-black ">
			<p>{text}</p>
			<div>
				<p>Category: {category}</p>
				<p>Intent: {intent}</p>
				<p>Named Entity Recognition: {ner}</p>
			</div>
			<div className="pb-2 flex justify-between items-center">
				{!isSelected && (
					<button
						onClick={onSelect}
						className="text-gray-900 rounded-lg text-center hover:text-gray-600 transition duration-150 font-bold"
					>
						Select this option &gt;
					</button>
				)}
				{isSelected && (
					<Link
						to={{
							pathname: `/result/`,
						}}
						className="text-gray-900 rounded-lg text-center hover:text-gray-600 transition duration-150 font-bold"
					>
						View More &gt;
					</Link>
				)}
			</div>
		</div>
	);
};

BotMessage.propTypes = {
	text: PropTypes.string.isRequired,
	category: PropTypes.string,
	intent: PropTypes.string,
	ner: PropTypes.string,
	id: PropTypes.number.isRequired,
	isSelected: PropTypes.bool.isRequired,
	onSelect: PropTypes.func.isRequired,
};

export default BotMessage;
