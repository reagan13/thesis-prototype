import PropTypes from "prop-types";

const UserMessage = ({ text }) => (
  <div className="text-right mt-2">
    <div className="border-r-[8px] border-[#00137F] bg-[#88D1FF] max-w-[400px] text-left text-black p-3 rounded-lg inline-block">
      {text}
    </div>
  </div>
);

UserMessage.propTypes = {
  text: PropTypes.string.isRequired, // Marking text as required
};

export default UserMessage;