import PropTypes from "prop-types";

const UserMessage = ({ text }) => (
	<div className="text-left mt-2 border border-black p-3 rounded-3xl max-w-[500px]  break-words">
		<p className="whitespace-pre-wrap">{text}</p>
	</div>
);

UserMessage.propTypes = {
	text: PropTypes.string.isRequired, // Marking text as required
};

export default UserMessage;
