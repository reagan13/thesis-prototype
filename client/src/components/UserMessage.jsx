import PropTypes from "prop-types";
const UserMessage = ({ text }) => (
  <div className="text-right mt-2">
    <div className="bg-blue max-w-[400px] text-left text-white p-3 rounded-lg inline-block">
      {text}
    </div>
  </div>
);
UserMessage.propTypes = {
  text: PropTypes.string,
};

export default UserMessage;
