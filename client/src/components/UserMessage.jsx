import PropTypes from "prop-types";
const UserMessage = ({ text }) => (
	<div className="text-right">
		<div className="bg-blue text-white p-2 rounded-lg inline-block">{text}</div>
	</div>
);
UserMessage.propTypes = {
	text: PropTypes.string,
};

export default UserMessage;
