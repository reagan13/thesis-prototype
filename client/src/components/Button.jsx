import PropTypes from "prop-types";

const Button = ({ label, onClick = () => {}, className = "" }) => {
	return (
		<button
			className={`bg-blue-500 text-white px-4 py-2 rounded ${className}`}
			onClick={onClick}
		>
			{label}
		</button>
	);
};

Button.propTypes = {
	label: PropTypes.string.isRequired,
	onClick: PropTypes.func,
	className: PropTypes.string,
};

export default Button;
