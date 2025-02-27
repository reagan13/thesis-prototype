import PropTypes from "prop-types";

const UserMessage = ({ text, timestamp }) => (
    <div className="text-left mt-2 border border-black p-3 rounded-3xl max-w-[500px] break-words">
        <p className="whitespace-pre-wrap">{text}</p>
        <p className="text-xs text-gray-500">{timestamp}</p> {/* Display timestamp */}
    </div>
);

UserMessage.propTypes = {
    text: PropTypes.string.isRequired,
    timestamp: PropTypes.string.isRequired,
};

export default UserMessage;