"""User management CLI for the Flask app.

Keeps backward compatibility with the old behavior:
  python create_admin.py
will ensure an admin user exists.

Now it also supports creating/updating any user:
  python create_admin.py --username alice --role user
  python create_admin.py --username bob --password "S3cret!" --role admin --reset-password

Bulk mode (JSON file with a list of users):
  python create_admin.py --users-file ./users.json --reset-password

Example users.json:
[
  {"username": "alice", "password": "...", "role": "user", "is_active": true},
  {"username": "bob",   "password": "...", "role": "admin"}
]
"""


# Crear usuario (te pedirá password por consola si no lo pasas):

# python create_admin.py --username valeria --role user

# Crear usuario con password en línea:

# python create_admin.py --username valeria --password "MiPass123!" --role user

# Resetear password de un usuario existente:

# python create_admin.py --username valeria --password "NuevaPass123!" --reset-password

# Crear varios usuarios con JSON:

# python create_admin.py --users-file users.json --reset-password

# Generar password aleatoria:

# python create_admin.py --username valeria --role user --generate-password

import argparse
import json
import secrets
import sys
from getpass import getpass
from typing import Any, Dict, Iterable, Optional

from app import create_app, db
from app.models import User


def _bool_from_any(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _generate_password(length: int = 18) -> str:
    # urlsafe already avoids problematic characters for copying.
    return secrets.token_urlsafe(max(12, length))


def upsert_user(
    *,
    username: str,
    password: Optional[str],
    role: str,
    is_active: bool,
    email: Optional[str] = None,
    reset_password: bool = False,
) -> Dict[str, Any]:
    """Create user if missing; update basic fields; optionally reset password."""
    username = (username or "").strip()
    if not username:
        raise ValueError("username is required")

    if role not in {"user", "admin"}:
        raise ValueError("role must be 'user' or 'admin'")

    user = User.query.filter_by(username=username).first()
    created = False

    if not user:
        user = User(username=username)
        db.session.add(user)
        created = True

    # Always keep these in sync with the requested state
    user.role = role
    user.is_active = bool(is_active)
    if email:
        user.email = email

    if created or reset_password:
        if not password:
            raise ValueError(f"password required for user '{username}' (or use --generate-password)")
        user.set_password(password)

    db.session.commit()
    return {"username": username, "created": created, "role": role, "is_active": is_active}


def _load_users_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("users-file JSON must be a list")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each user item in users-file must be an object")
        yield item


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Create/update users for the app")

    parser.add_argument("--username", help="Username to create/update")
    parser.add_argument("--password", help="Password to set (use with --reset-password or for new user)")
    parser.add_argument("--email", help="Email (optional)")
    parser.add_argument("--role", default=None, choices=["user", "admin"], help="Role")
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Create/update user as inactive (default: active)",
    )
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="Reset password if user already exists",
    )
    parser.add_argument(
        "--generate-password",
        action="store_true",
        help="Generate a random password (printed once)",
    )
    parser.add_argument(
        "--users-file",
        help="Path to JSON file with a list of users for bulk creation/update",
    )

    args = parser.parse_args(argv)

    app = create_app()
    with app.app_context():
        # Backward compatible mode: ensure admin exists.
        if not args.username and not args.users_file:
            username = "admin"
            password = "admin1234"
            user = User.query.filter_by(username=username).first()
            if not user:
                user = User(username=username, role="admin", is_active=True)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                print("✅ Admin creado:", username, password)
            else:
                print("ℹ️ Admin ya existe:", username)
            return 0

        # Bulk mode
        if args.users_file:
            results = []
            for u in _load_users_file(args.users_file):
                username = (u.get("username") or "").strip()
                role = (u.get("role") or args.role or "user").strip()
                is_active = _bool_from_any(u.get("is_active"), default=not args.inactive)
                email = u.get("email") or None

                password = u.get("password") or args.password
                generated_password = None
                if not password and args.generate_password:
                    generated_password = _generate_password()
                    password = generated_password
                if not password:
                    raise ValueError(
                        f"Missing password for '{username}'. Provide it in JSON or use --generate-password"
                    )

                res = upsert_user(
                    username=username,
                    password=password,
                    role=role,
                    is_active=is_active,
                    email=email,
                    reset_password=args.reset_password,
                )
                res["password"] = generated_password  # printed only if generated
                results.append(res)

            print(f"✅ Usuarios procesados: {len(results)}")
            for r in results:
                extra = ""
                if r.get("password"):
                    extra = f" | password generado: {r['password']}"
                print(
                    f" - {r['username']} | {'CREADO' if r['created'] else 'ACTUALIZADO'} | role={r['role']} | active={r['is_active']}{extra}"
                )
            return 0

        # Single user mode
        username = (args.username or "").strip()
        if not username:
            raise ValueError("--username is required (or use --users-file)")

        role = (args.role or "user").strip()
        is_active = not args.inactive

        existing = User.query.filter_by(username=username).first()

        password = args.password
        generated_password = None
        if not password and args.generate_password:
            generated_password = _generate_password()
            password = generated_password

        # Password handling rules:
        # - New user: password required (prompt if not provided)
        # - Existing user: password required only if --reset-password

        def _prompt_password(label: str) -> str:
            # getpass oculta lo que se teclea (no verás caracteres). En algunos terminals puede fallar.
            if not sys.stdin.isatty():
                raise ValueError(
                    "No hay terminal interactiva (stdin no es TTY). Usa --password o --generate-password."
                )
            try:
                print("(La contraseña no se mostrará mientras se escribe. Pulsa Enter al terminar.)")
                return getpass(label)
            except Exception:
                # Fallback para terminals que no soportan getpass bien (p.ej. algunos integrados)
                return input(label.replace(":", " (visible): "))

        if not existing and not password:
            password = _prompt_password("Password: ")
        if existing and args.reset_password and not password:
            password = _prompt_password("New password: ")


        if existing and not args.reset_password:
            # Don't touch the password.
            password = None

        res = upsert_user(
            username=username,
            password=password,
            role=role,
            is_active=is_active,
            email=args.email,
            reset_password=args.reset_password,
        )

        print(
            f"✅ {res['username']} | {'CREADO' if res['created'] else 'ACTUALIZADO'} | role={res['role']} | active={res['is_active']}"
        )
        if generated_password:
            print("🔑 Password generado (guárdalo ahora, no se volverá a mostrar):", generated_password)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
